from typing import Optional
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataset_utils import ColorizationFolderDataset
from loss_functions import GANLoss
from critics import Discriminator
from generators import UNet
from tqdm import tqdm

CIE_LAB = True
BATCH_SIZE = 16
TRAIN_FOLDER = 'data/imagenet_tiny/train'
MODEL_PATH = 'output/model.pth'


class Model(nn.Module):
    def __init__(self, net_G: nn.Module, net_D: nn.Module,
                 lr_G: float = 2e-4, lr_D: float = 2e-4,
                 lambda_L1: float = 25.):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        self.net_G = net_G.to(self.device)
        self.net_D = net_D.to(self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()

        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G)
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D)
        # self.scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.opt_G, mode='min', factor=0.3, patience=3, verbose=True)
        # self.scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.opt_D, mode='min', factor=0.3, patience=3, verbose=True)

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, L, Xn, ab):
        self.L = L.to(self.device)
        self.Xn = Xn.to(self.device)
        self.ab = ab.to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.Xn)

    def get_loss_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D
        # self.loss_D.backward()

    def get_loss_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(
            self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        return self.loss_G
        # self.loss_G.backward()

    def optimize(self, only_disc: bool = True):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.get_loss_D().backward()
        self.opt_D.step()
        # self.scheduler_D.step(self.loss_D.item())

        if not only_disc:
            self.net_G.train()
            self.set_requires_grad(self.net_D, False)
            self.opt_G.zero_grad()
            self.get_loss_G().backward()
            self.opt_G.step()
            # self.scheduler_G.step(self.loss_G.item())


def train(model: nn.Module, train_dataloader: DataLoader,
          val_dataloader: Optional[nn.Module] = None,
          eval_every: int = 50, total_iterations: int = 1000,
          only_disc: bool = True):

    cur_iter = 0
    cur_loss_G = 0.0
    cur_loss_D = 0.0
    cur_samples = 0
    train_loss_G = []
    train_loss_D = []
    val_loss_G = []
    val_loss_D = []

    pbar = tqdm(total=eval_every)
    while True:
        for L, Xn, ab in train_dataloader:

            model.setup_input(L, Xn, ab)
            model.optimize(only_disc)
            cur_loss_D += model.loss_D.item() * len(L)
            if not only_disc:
                cur_loss_G += model.loss_G.item() * len(L)
            cur_samples += len(L)
            pbar.update(1)
            cur_iter += 1
            # actions for current 'eval every'
            if cur_iter % eval_every == 0:
                train_loss_D.append(cur_loss_D / cur_samples)
                print(f'\n  Discriminator train loss: {train_loss_D[-1]:.2e}')
                cur_loss_D = 0.0
                if not only_disc:
                    train_loss_G.append(cur_loss_G / cur_samples)
                    print(f'\n  Generator train loss: {train_loss_G[-1]:.2e}')
                    cur_loss_G = 0.0
                cur_samples = 0
                pbar.close()
                pbar = tqdm(total=eval_every)
            if cur_iter >= total_iterations:
                pbar.close()
                return (np.array(train_loss_D), np.array(val_loss_D),
                        np.array(train_loss_G), np.array(val_loss_G))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_G = UNet(
            resnet_layers=18,
            cie_lab=CIE_LAB
        ).to(device).eval()
    net_G.load_state_dict(
        torch.load(MODEL_PATH, map_location=device))
    net_D = Discriminator()

    train_dataset = ColorizationFolderDataset(
            folder=TRAIN_FOLDER,
            transforms=T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
            ]),
            cie_lab=CIE_LAB
        )

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = Model(net_G, net_D)
    train(model, train_dataloader)


if __name__ == '__main__':
    main()
