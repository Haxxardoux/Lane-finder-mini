import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from VQVAE import VQVAE_Encoder as small_model
from VQVAE import VQVAE as big_model
from train import train, knowledge_distillation
from utilities import start_mlflow_experiment, Params, save_to_mlflow, count_parameters, load_full_state, select_gpu

from tqdm import tqdm
import mlflow

args = Params(16, 10, 4e-4, 256, 'cuda:0')

start_mlflow_experiment('VQVAE2 Knowledge distillation', 'lane-finder')


transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

dataset = datasets.ImageFolder('/share/lazy/will/ConstrastiveLoss/Imgs/color_images/train/', transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory = True)

device = 'cuda:1' 
teacher_model = big_model(channel=128).to(device)

# optimizer declaration does nothing
optimizer = optim.Adam(teacher_model.parameters(), lr=args.lr)
load_full_state(teacher_model, optimizer, '/share/lazy/will/ConstrastiveLoss/Logs/0/64a43ca191944cba89536145c4422027/artifacts/run_stats.pyt', freeze_weights=False)


device = 'cuda:1' 

stuff_to_try = [
    (Params(16, 1, 4e-4, 256, device), big_model(n_res_block=0, n_extra_layers=1).to(device)),
    (Params(16, 1, 4e-4, 256, device), big_model(n_res_block=0, n_extra_layers=0).to(device))
]
     
for args, student_model in stuff_to_try:

    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)

    run_name = 'overnight run'

    with mlflow.start_run(run_name = run_name) as run:

        for epoch in range(args.epoch):
            results = train(epoch, loader, student_model, optimizer, args.device)
            for Dict in results:
                save_to_mlflow(Dict, args)
