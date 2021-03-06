from typing import Iterable, Optional
import numpy as np
import torch
from torch import nn
import torchvision as tv


class VGG16Loss(nn.Module):
    LAYER_NAMES = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'maxpool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'maxpool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'maxpool5'
    )

    def __init__(self,
                 feat_layers: Iterable[str], feat_weights: Iterable[float],
                 style_layers: Iterable[str], style_weights: Iterable[float],
                 base_loss: Optional[nn.Module] = nn.MSELoss):
        super().__init__()
        # load pretrained VGG16
        self.vgg = tv.models.vgg16(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        # store feature and style loss weights for every VGG16 layer
        self.fw = np.zeros(len(self.LAYER_NAMES))   # feature loss weights
        self.sw = np.zeros_like(self.fw)            # style loss weights
        for name, weight in zip(feat_layers, feat_weights):
            layer_ind = self.LAYER_NAMES.index(name)
            self.fw[layer_ind] = weight
        for name, weight in zip(style_layers, style_weights):
            layer_ind = self.LAYER_NAMES.index(name)
            self.sw[layer_ind] = weight

        self.normalize = tv.transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.loss_fn = base_loss()

    def _style_loss(self, x: torch.tensor,
                    target: torch.tensor) -> torch.tensor:
        Gx = gram_matrix(x)
        Gy = gram_matrix(target)
        return self.loss_fn(Gx, Gy)

    def _feature_loss(self, x: torch.tensor,
                      target: torch.tensor) -> torch.tensor:
        return self.loss_fn(x, target)

    def forward(self, x: torch.tensor, target: torch.tensor) -> torch.tensor:
        x = self.normalize(x)
        y = target.detach()
        y = self.normalize(y)
        loss = 0.0
        for i, layer in enumerate(self.vgg.children()):
            x = layer(x)
            y = layer(y)
            if self.fw[i] > 0:
                loss += self.fw[i] * self._feature_loss(x, y)
            if self.sw[i] > 0:
                loss += self.sw[i] * self._style_loss(x, y)
        return loss


def gram_matrix(x: torch.tensor) -> torch.tensor:
    # from "Neural Style Transfer Using PyTorch" tutorial
    a, b, c, d = x.size()
    features = x.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class GANLoss(nn.Module):
    def __init__(self, device: torch.device, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label, device=device))
        self.register_buffer('fake_label', torch.tensor(fake_label, device=device))
        if gan_mode == 'vanilla':
            self.loss = nn.BCELoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss