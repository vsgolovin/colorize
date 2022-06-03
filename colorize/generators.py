import torch
from torch import nn
import torchvision


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # get pretrained ResNet backbone to use as an encoder
        resnet = torchvision.models.resnet34(pretrained=True)
        resnet.eval()
        for param in resnet.parameters():
            param.requires_grad = False

        # encoder
        self.enc_block1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )  # -> (64, 112, 112)
        self.enc_block2 = nn.Sequential(
            resnet.maxpool, resnet.layer1
        )  # -> (64, 56, 56)
        self.enc_block3 = resnet.layer2  # -> (128, 28, 28)
        self.enc_block4 = resnet.layer3  # -> (256, 14, 14)
        self.enc_block5 = resnet.layer4  # -> (512, 7, 7)

        # decoder
        self.dec_block5 = self._dec_block(512, False)  # -> (256, 14, 14)
        self.dec_block4 = self._dec_block(512, True)  # -> (128, 28, 28)
        self.dec_block3 = self._dec_block(256, True)  # -> (64, 56, 56)
        self.dec_block2 = self._dec_block(128, False)  # -> (64, 112, 112)
        self.dec_block1 = self._dec_block(128, True)  # -> (32, 224, 224)

        self.output_conv = nn.Sequential(
            nn.Conv2d(32, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, (3, 3), stride=1, padding=1),
        )  # -> (3, 224, 224)

    def forward(self, X: torch.tensor) -> torch.tensor:
        features = []
        for module in (
            self.enc_block1,
            self.enc_block2,
            self.enc_block3,
            self.enc_block4,
        ):
            X = module(X)
            features.append(X.clone())
        X = self.dec_block5(self.enc_block5(X))
        for feature, module in zip(
            reversed(features),
            (self.dec_block4, self.dec_block3, self.dec_block2, self.dec_block1),
        ):
            X = torch.cat([feature, X], dim=1)
            X = module(X)
        return self.output_conv(X)

    @staticmethod
    def _dec_block(c_in: int, decrease_twice: bool) -> nn.Module:
        k = 2 if decrease_twice else 1
        nc = (c_in, c_in // k, c_in // 2 // k)  # number of channels
        return nn.Sequential(
            nn.Conv2d(nc[0], nc[1], (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(nc[1], nc[2], (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )
