import torch
from torch import nn, Tensor
from torchvision import models
import layers


class UNet(nn.Module):
    def __init__(self, resnet: models.ResNet, self_attention: bool = False,
                 blur: bool = False, blur_final: bool = True, **kwargs):
        super().__init__()
        assert 'self_attention' not in kwargs

        # encoder
        # comments show output shape given (3, 224, 224) input
        self.enc_block1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu
                                        )  # -> (64, 112, 112)
        self.enc_block2 = nn.Sequential(resnet.maxpool, resnet.layer1
                                        )  # -> (64, 56, 56)
        self.enc_block3 = resnet.layer2    # -> (128, 28, 28)
        self.enc_block4 = resnet.layer3    # -> (256, 14, 14)
        self.enc_block5 = resnet.layer4    # -> (512, 7, 7)

        # decoder
        self.dec_block5 = self._dec_block((512, 1024, 512), 256, blur=blur,
                                          **kwargs)  # -> (256, 14, 14)
        self.dec_block4 = self._dec_block((512, 512, 512), 256, blur=blur,
                                          **kwargs)  # -> (256, 28, 28)
        self.dec_block3 = self._dec_block((384, 384, 384), 192, blur=blur,
                                          self_attention=self_attention,
                                          **kwargs)  # -> (192, 56, 56)
        self.dec_block2 = self._dec_block((256, 256, 256), 128, blur=blur,
                                          **kwargs)  # -> (128, 112, 112)
        self.dec_block1 = self._dec_block((192, 96, 96), 96,
                                          blur=blur and blur_final,
                                          **kwargs)  # -> (96, 224, 224)

        # final convolution
        self.output_conv = nn.Sequential(
            layers.ResidualUnit33((99, 99, 99), **kwargs),
            nn.Conv2d(99, 3, 1, 1, 0),
            # nn.Sigmoid()
        )  # -> (3, 224, 224)

    # freezing / unfreezing encoder weights
    def freeze_encoder(self):
        self._set_encoder_grad(False)

    def unfreeze_encoder(self):
        self._set_encoder_grad(True)

    def forward(self, X: Tensor) -> Tensor:
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

    def _set_encoder_grad(self, value: bool):
        for module in (self.enc_block1, self.enc_block2, self.enc_block3,
                       self.enc_block4, self.enc_block5):
            for param in module.parameters():
                param.requires_grad = value

    # convolution + upsample
    def _dec_block(self, c_conv: tuple[int], c_out: int, blur: bool,
                   self_attention: bool = False, **kwargs) -> nn.Module:
        if len(c_conv) == 3:
            conv = layers.ConvBlock33
        else:
            assert len(c_conv) == 4
            conv = layers.ConvBlock131
        return nn.Sequential(
            conv(c_conv, self_attention=self_attention, **kwargs),
            layers.PixelShuffle_ICNR(c_conv[-1], c_out*4, scale=2, blur=blur),
        )
