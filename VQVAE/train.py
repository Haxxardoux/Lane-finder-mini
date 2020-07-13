import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utilities import count_parameters
from torchvision import datasets, transforms, utils

from tqdm import tqdm

def train(epoch, loader, model, optimizer, device):
    '''
    params: epoch, loader, model, optimizer, device
    checkpoint gets saved to "run_stats.pyt"
    '''
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0
    params = count_parameters(model)
    for i, (img, labels) in enumerate(loader):
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

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                f"lr: {lr:.5f}"
            )
        )

        if i % 100 == 0:
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
            torch.save({
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            }, 'run_stats.pyt')
            model.train()

        ret = {'Metric: Latent Loss':latent_loss.item(), 'Metric: Average MSE':mse_sum/mse_n, 'Metric: Reconstruction Loss':recon_loss.item(), 'Parameter: Parameters':params, 'Artifact':'run_stats.pyt'}
        yield ret

        
def knowledge_distillation(epoch, loader, teacher_model, student_model, optimizer, device):
    '''
    epoch, loader, teacher_model, student_model, optimizer, device
    '''
    loader = tqdm(loader)

    criterion = nn.MSELoss()
    
    mse_sum = 0
    mse_n = 0
    
    t_params = count_parameters(teacher_model)
    s_params = count_parameters(student_model)
    
    for i, (img, labels) in enumerate(loader):
        teacher_model.zero_grad()
        teacher_model.eval()
        student_model.zero_grad()

        img = img.to(device)

        teacher_quant_t, teacher_quant_b, _, t_id_t, t_id_b = teacher_model.encode(img)
        student_quant_t, student_quant_b, _, s_id_t, s_id_b = student_model.encode(img)

        t_mse = criterion(student_quant_t, teacher_quant_t)
        b_mse = criterion(student_quant_b, teacher_quant_b)

        loss = t_mse + b_mse
        loss.backward()

        optimizer.step()

        mse_sum += loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"epoch: {epoch + 1}; avg mse: {mse_sum / mse_n:.5f}; "
                f"lr: {lr:.5f}"
            )
        )

        if i % 100 == 0:
            student_model.eval()

            sample_size = 10
            sample = img[:sample_size]

            with torch.no_grad():
                out = teacher_model.decode(student_quant_t, student_quant_b)

            utils.save_image(
                torch.cat([sample, out], 0),
                f"samples/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg",
                nrow=sample_size,
                normalize=True,
                range=(-1, 1))

            torch.save({
            'model':student_model.state_dict(),
            'optimizer':optimizer.state_dict(),
            }, 'run_stats.pyt')

            student_model.train()

        ret = {'Metric: Average MSE':mse_sum/mse_n, 'Parameter: Student Parameters':s_params, 'Parameter: Teacher Parameters':t_params, 'Artifact':'run_stats.pyt'}
        yield ret