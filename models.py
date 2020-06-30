import torch.nn as nn
import torch
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__()
        self.residualBlock = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
        nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return nn.ReLU()(self.residualBlock(x) + x)

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
        DoubleConv(in_channels, out_channels),
        nn.MaxPool2d(2))

class Backbone(nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.n = n
        self.start = nn.Conv2d(3, self.n, kernel_size=3, padding=1)
        self.C1 = DoubleConv(self.n, self.n)
        self.C2 = DoubleConv(self.n, self.n)
        self.C3 = DoubleConv(self.n, self.n)
        self.down1 = Down(self.n, self.n)
        self.C4 = DoubleConv(self.n, self.n)
        self.C5 = DoubleConv(self.n, self.n)
        self.C6 = DoubleConv(self.n, self.n)
        self.down2 = Down(self.n, self.n)
        self.C7 = DoubleConv(self.n, self.n)
        self.C8 = DoubleConv(self.n, self.n)
        self.C9 = DoubleConv(self.n, self.n)
        self.down3 = Down(self.n, self.n)
        self.C10 = DoubleConv(self.n, self.n)
        self.C11 = DoubleConv(self.n, self.n)
        self.C12 = DoubleConv(self.n, self.n)
        self.down4 = Down(self.n, self.n)
        self.C10 = DoubleConv(self.n, self.n)
        self.C11 = DoubleConv(self.n, self.n)
        self.C12 = DoubleConv(self.n, self.n)
        self.down5 = Down(self.n, self.n)
        self.C13 = DoubleConv(self.n, self.n)
        self.C14 = DoubleConv(self.n, self.n)
        self.C15 = DoubleConv(self.n, self.n)
        self.down6 = Down(self.n, self.n//2)
        # flatten op
        self.L1 = nn.Linear(2048, n*2)
        self.L2 = nn.Linear(n*2, n*2)        
        self.L3 = nn.Linear(n*2, n*2)
        self.r = nn.ReLU()


    def forward(self, x):
        x = self.start(x)
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.down1(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)
        x = self.down2(x)
        x = self.C7(x)
        x = self.C8(x)
        x = self.C9(x)
        x = self.down3(x)
        x = self.C10(x)
        x = self.C11(x)
        x = self.C12(x)
        x = self.down4(x)
        x = self.C10(x)
        x = self.C11(x)
        x = self.C12(x)
        x = self.down5(x)
        x = self.C13(x)
        x = self.C14(x)
        x = self.C15(x)
        x = self.down5(x)
        x = nn.Flatten()(x)
        x = self.L1(x)
        x = self.r(x)
        x = self.L2(x)
        x = self.r(x)
        x = self.L3(x)
        x = self.r(x)

        return x

    
class BackboneSmaller(nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.n = n
        self.start = nn.Conv2d(3, self.n, kernel_size=3, padding=1)
        self.C1 = DoubleConv(self.n, self.n)
        self.down1 = Down(self.n, self.n)
        self.C4 = DoubleConv(self.n, self.n)
        self.down2 = Down(self.n, self.n)
        self.C7 = DoubleConv(self.n, self.n)
        self.down3 = Down(self.n, self.n)
        self.C12 = DoubleConv(self.n, self.n)
        self.down4 = Down(self.n, self.n)
        self.C12 = DoubleConv(self.n, self.n)
        self.down5 = Down(self.n, self.n)
        self.C15 = DoubleConv(self.n, self.n)
        self.down6 = Down(self.n, self.n//2)
        # flatten op
        self.L1 = nn.Linear(2048, n*2)
        self.L2 = nn.Linear(n*2, n*2)        
        self.L3 = nn.Linear(n*2, n*2)
        self.r = nn.ReLU()


    def forward(self, x):
        x = self.start(x)
        x = self.C1(x)
        x = self.down1(x)
        x = self.C4(x)
        x = self.down2(x)
        x = self.C7(x)
        x = self.down3(x)
        x = self.C12(x)
        x = self.down4(x)
        x = self.C12(x)
        x = self.down5(x)
        x = self.C15(x)
        x = self.down5(x)
        x = nn.Flatten()(x)
        x = self.L1(x)
        x = self.r(x)
        x = self.L2(x)
        x = self.r(x)
        x = self.L3(x)
        x = self.r(x)

        return x
    
    
    
import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn



# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


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

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

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
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

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
    from torchsummary import summary
    summary(VQVAE().to('cuda:0'), (3, 288, 512))

