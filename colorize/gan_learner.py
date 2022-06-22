from typing import Optional, Union, Callable
import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataset_utils import ColorizationFolderDataset
from loss_functions import GANLoss
from critics import SimpleCritic
from generators import UNet
from tqdm import tqdm

CIE_LAB = True
BATCH_SIZE = 16
TRAIN_FOLDER = 'data/imagenet_tiny/train'
MODEL_PATH = 'checkpoints/resnet18.pth'


class GANLearner:
    default_optG_settings = dict(lr=1e-4, betas=(0.5, 0.99))
    default_optD_settings = dict(lr=2e-4, betas=(0.5, 0.99))

    def __init__(self, net_G: nn.Module, net_D: nn.Module,
                 pixel_loss: Callable = F.l1_loss,
                 pixel_loss_weight: float = 25.,
                 gan_mode: str = 'vanilla',
                 gen_opt_params: Optional[dict] = None,
                 discr_opt_params: Optional[dict] = None,
                 device: Union[str, torch.device] = 'cpu'):
        super().__init__()
        # device to use
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # loss functions
        self.pixel_loss = pixel_loss
        self.pixel_loss_weight = pixel_loss_weight
        self.gan_loss = GANLoss(self.device, gan_mode=gan_mode)

        # generator and discriminator networks
        self.net_G = net_G.to(self.device)
        self.net_D = net_D.to(self.device)

        # optimizers
        if gen_opt_params is None:
            gen_opt_params = self.default_optG_settings
        self.opt_G = optim.Adam(self.net_G.parameters(), **gen_opt_params)
        if discr_opt_params is None:
            discr_opt_params = self.default_optD_settings
        self.opt_D = optim.Adam(self.net_D.parameters(), **discr_opt_params)

    # freezing / unfreezing generator for discriminator pretraining
    def freeze_generator(self):
        self.net_G.eval()
        self._set_requires_grad(self.net_G, False)

    def unfreeze_generator(self):
        self.net_G.train()
        self._set_requires_grad(self.net_G, True)

    @staticmethod
    def _set_requires_grad(model: nn.Module, value: bool):
        for param in model.parameters():
            param.requires_grad = value

    def _make_images(self, L: Tensor, Ln: Tensor, AB: Tensor
                     ) -> tuple[Tensor, Tensor]:
        AB_fake = self.net_G(Ln)
        fake_imgs = torch.cat([L, AB_fake], dim=1)
        real_imgs = torch.cat([L, AB], dim=1)
        return fake_imgs, real_imgs

    def train_iter(self, batch: tuple[Tensor], discr_only: bool = True):
        self.net_D.train()
        if discr_only:
            self.net_G.eval()
        else:
            self.net_G.train()

        L, Xn, AB = [t.to(self.device) for t in batch]
        fake_imgs, real_imgs = self._make_images(L, Xn, AB)

        # ree
        probs_fake = self.net_D(fake_imgs.detach())
        probs_real = self.net_D(real_imgs)
        lossD_fake = self.gan_loss(probs_fake, False)
        lossD_real = self.gan_loss(probs_real, True)
        lossD = (lossD_fake + lossD_real) / 2
        self.opt_D.zero_grad()
        lossD.backward()
        self.opt_D.step()

        if not discr_only:
            probs = self.net_D(fake_imgs)
            lossG_gan = self.gan_loss(probs, True)
            lossG_pixel = self.pixel_loss(fake_imgs, real_imgs)
            lossG = lossG_gan + lossG_pixel * self.pixel_loss_weight
            self.opt_G.zero_grad()
            lossG.backward()
            self.opt_G.step()
        else:
            lossG = torch.tensor([0.0])

        return lossD.item(), lossG.item()


def train(model: GANLearner, train_dataloader: DataLoader,
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
        for batch in train_dataloader:
            bs = len(batch[0])
            lossD, _ = model.train_iter(batch, True)
            cur_loss_D += lossD * bs
            # if not only_disc:
            #     cur_loss_G += model.loss_G.item() * len(L)
            cur_samples += bs
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
    net_D = SimpleCritic()

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
    model = GANLearner(net_G, net_D, device='cuda')
    train(model, train_dataloader)


if __name__ == '__main__':
    main()
