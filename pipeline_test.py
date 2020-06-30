import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm
import mlflow

from VQVAE import VQVAE
import distributed as dist

class Params(object):
    def __init__(self, batch_size, epochs, lr, size):
        self.size = batch_size
        self.epoch = epochs
        self.lr = lr
        self.size = size

args = Params(64, 1000, 0.0005, 256)

def train(epoch, loader, model, optimizer, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )
            
        if i % 20 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f"samples/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg",
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()

        return {'Latent Loss':latent_loss.item(), 'Average MSE':mse_sum/mse_n, 'Reconstruction Loss':recon_loss.item()}

    
device = "cuda:0"

mlflow.tracking.set_tracking_uri('file:/share/lazy/will/ConstrastiveLoss/Logs')

mlflow.set_experiment('Vector Quantized Variational Autoencider')

transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

dataset = datasets.ImageFolder('/data/home/will/Lane-finder-mini/', transform=transform)
# sampler = dist.data_sampler(dataset, shuffle=True, distributed=False)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = VQVAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

run_name = 'VQVAE 2 on comma10k (3k)'
with mlflow.start_run(run_name = run_name) as run:

    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    mlflow.log_param('Parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch in range(args.epoch):
        results = train(epoch, loader, model, optimizer, device)
        for key, value in results.items():
            print(key, value)
            mlflow.log_metric(key, value, epoch)

            torch.save({
    'model':model.state_dict(),
    'optimizer':optimizer.state_dict(),
    }, 'run_stats.pyt')
    mlflow.log_artifact('run_stats.pyt')

