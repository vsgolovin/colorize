from typing import Optional, Union, Callable
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from colorize.loss_functions import GANLoss


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
        self.net_D.train()  # never eval()!

        # optimizers
        if gen_opt_params is None:
            gen_opt_params = self.default_optG_settings
        self.opt_G = optim.Adam(self.net_G.parameters(), **gen_opt_params)
        if discr_opt_params is None:
            discr_opt_params = self.default_optD_settings
        self.opt_D = optim.Adam(self.net_D.parameters(), **discr_opt_params)

        # save last generated images
        self.fake_images = None
        self.real_images = None

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
        self.fake_images = fake_imgs.cpu().detach()
        self.real_images = real_imgs.cpu().detach()
        return fake_imgs, real_imgs

    def iteration(self, batch: tuple[Tensor], discr_only: bool = True,
                  train: bool = True) -> tuple[float, float]:
        if not train or discr_only:
            self.net_G.eval()
        else:
            self.net_G.train()

        # generate images and check them with discriminator
        L, Xn, AB = [t.to(self.device) for t in batch]
        fake_imgs, real_imgs = self._make_images(L, Xn, AB)
        probs_fake = self.net_D(fake_imgs.detach())
        probs_real = self.net_D(real_imgs)
        correct_predictions = ((probs_fake < 0.5).sum()
                               + (probs_real >= 0.5).sum()).item()

        # update discriminator
        lossD_fake = self.gan_loss(probs_fake, False)
        lossD_real = self.gan_loss(probs_real, True)
        lossD = (lossD_fake + lossD_real) / 2
        if train:
            self.opt_D.zero_grad()
            lossD.backward()
            self.opt_D.step()

        # update generator
        if not discr_only:
            probs = self.net_D(fake_imgs)
            lossG_gan = self.gan_loss(probs, True)
            lossG_pixel = self.pixel_loss(fake_imgs, real_imgs)
            lossG = lossG_gan + lossG_pixel * self.pixel_loss_weight
            if train:
                self.opt_G.zero_grad()
                lossG.backward()
                self.opt_G.step()
        else:
            lossG = torch.tensor([0.0])

        return lossD.item(), lossG.item(), correct_predictions

    def train_iter(self, batch: tuple[Tensor], discr_only: bool = True
                   ) -> tuple[float, float]:
        """
        Perform single training iteration.
        """
        return self.iteration(batch, discr_only, True)

    @torch.no_grad()
    def eval_iter(self, batch: tuple[Tensor], discr_only: bool = True
                  ) -> tuple[float, float]:
        """
        Evaluate model for a single batch.
        """
        return self.iteration(batch, discr_only, False)
