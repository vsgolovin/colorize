import torch
from torch import nn
import torchvision
import layers


class UNet34(nn.Module):
    def __init__(self):
        super().__init__()

        # get pretrained ResNet backbone to use as an encoder
        resnet = torchvision.models.resnet34(pretrained=True).eval()
        for param in resnet.parameters():
            param.requires_grad = False

        # encoder
        self.enc_block1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu
                                        )  # -> (64, 112, 112)
        self.enc_block2 = nn.Sequential(resnet.maxpool, resnet.layer1
                                        )  # -> (64, 56, 56)
        self.enc_block3 = resnet.layer2    # -> (128, 28, 28)
        self.enc_block4 = resnet.layer3    # -> (256, 14, 14)
        self.enc_block5 = resnet.layer4    # -> (512, 7, 7)

        # decoder
        self.dec_block5 = self._dec_block((512, 1024, 512), 256
                                          )  # -> (256, 14, 14)
        self.dec_block4 = self._dec_block((512, 768, 768), 384
                                          )  # -> (384, 28, 28)
        self.dec_block3 = self._dec_block((512, 768, 768), 384,
                                          self_attention=True
                                          )  # -> (384, 56, 56)
        self.dec_block2 = self._dec_block((448, 672, 672), 336,
                                          )  # -> (336, 112, 112)
        self.dec_block1 = self._dec_block((400, 300, 300), 300,
                                          )  # -> (300, 224, 224)

        # final convolution
        self.output_conv = nn.Sequential(
            layers.ConvBlock33((303, 303, 303)),
            nn.Conv2d(303, 3, 1, 1, 0),
            nn.Sigmoid()
        )  # -> (3, 224, 224)

    def forward(self, X: torch.tensor) -> torch.tensor:
        inp = X.clone()
        features = []
        for module in (self.enc_block1, self.enc_block2,
                       self.enc_block3, self.enc_block4):
            X = module(X)
            features.append(X.clone())
        X = self.dec_block5(self.enc_block5(X))
        for feature, module in zip(reversed(features),
                                   (self.dec_block4, self.dec_block3,
                                    self.dec_block2, self.dec_block1)):
            X = torch.cat([feature, X], dim=1)
            X = module(X)
        return self.output_conv(torch.cat([inp, X], dim=1))

    @staticmethod
    def _dec_block(c_conv: tuple[int], c_out: int, **kwargs) -> nn.Module:
        return nn.Sequential(
            layers.ConvBlock33(c_conv, kaiming_init=True, spectral_norm=True,
                               **kwargs),
            layers.PixelShuffle_ICNR(c_conv[-1], c_out*4, scale=2, blur=True),
        )
