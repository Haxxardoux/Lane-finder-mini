import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
import mlflow
from torchvision import transforms

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            torch.sum(embed_onehot_sum)
            torch.sum(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3, extra_layers=1, residual=True):
        super().__init__()
        self.residual=residual
        
        layers = [
            nn.Conv2d(in_channel, out_channel, stride=stride, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()]
        
        extra_block = [
            nn.Conv2d(out_channel, out_channel, stride=1, kernel_size=3, padding=(3-1)//2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()]

        layers.extend(extra_block)

        self.resblock = nn.Sequential(*layers)

    def forward(self, input):
        if self.residual:
            out = self.resblock(input)
            out = input+out
            return out
        else:
            out = self.resblock(input)
            return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, extra_layers, stride, kernel_size, residual, extra_residual_blocks, downsample):
        super().__init__()

        self.out_channels = channel

        blocks = [
            ResBlock(in_channel, channel, extra_layers=extra_layers, stride=stride, residual=residual),
            nn.ReLU(inplace=True)
        ]


        for i in range(extra_residual_blocks):
            blocks.append(ResBlock(in_channel=channel, out_channel=channel, extra_layers=extra_layers, residual=True))
            if (downsample=='Once') & (i==0):
                blocks.append(nn.MaxPool2d(2, 2))
            if (downsample=='Twice') & ((i==0) | (i==1)):
                blocks.append(nn.MaxPool2d(2, 2))

        self.encode = nn.Sequential(*blocks)

    def forward(self, input):
        return self.encode(input)


class Decoder(nn.Module):
    def __init__(self, channel, out_channel, extra_layers, extra_residual_blocks, upsample):
        super().__init__()

        blocks = []

        for i in range(extra_residual_blocks):
            blocks.append(ResBlock(in_channel=channel, out_channel=channel, extra_layers=extra_layers, residual=True))
            if (upsample=='Twice') & (i==0):
                blocks.append(nn.ConvTranspose2d(channel, channel, 2, 2))
                            
        blocks.append(nn.ConvTranspose2d(channel, out_channel, 2, 2))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    '''
    params: in_channel=3, channel=64, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99
    '''
    def __init__(
        self,
        in_channel=3,
        channel=64,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99
    ):
        '''
        params: in_channel=3, channel=64, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99
        '''
        super().__init__()
        # Encoders, first one should have two rounds of downsampling, second should have one
        self.enc_b = Encoder(in_channel=in_channel, channel=channel, extra_layers=2, stride=2, kernel_size=5, residual=False, extra_residual_blocks=2, downsample='Once')
        self.enc_t = Encoder(in_channel=channel, channel=channel, extra_layers=3, stride=1, kernel_size=3, residual=False, extra_residual_blocks=2, downsample='Once')

        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)

        # Decoders,
#         self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, extra_residual_blocks = n_res_block, upsample='Once')
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
#         self.dec = Decoder(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.dec = Decoder(embed_dim + embed_dim, in_channel, extra_layers=2, extra_residual_blocks=2, upsample='Twice')

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

if __name__ == "__main__":
    model = VQVAE().to('cuda:0')
    summary(model, (3, 256, 256))
    # model(torch.ones(1,3,256,256).to('cuda:0'))[0].shape
    
    thing1 = Encoder(in_channel=3, channel=128, extra_layers=2, stride=2, kernel_size=5, residual=False, extra_residual_blocks=2, downsample='Once')
    thing2 = Encoder(in_channel=thing1.out_channels, channel=64, extra_layers=3, stride=1, kernel_size=3, residual=False, extra_residual_blocks=2, downsample='Once')
    thing3 = Decoder(64, 32, extra_layers=2, extra_residual_blocks=2, upsample='Twice')
    
    summary(thing1.to('cuda:0'), (3, 256, 256))
    # summary(thing2.to('cuda:0'), (thing1.out_channels, 64, 64))
    # summary(thing3.to('cuda:0'), (64, 32, 32))
