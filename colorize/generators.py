from typing import Sequence
import torch
from torch import nn, Tensor
from torchvision import models
import layers


# small models
CONV_CHANNELS_S = (
    (64, 64, 64),     # output residual block
    (128, 64, 64),    # decoder block #1
    (128, 128, 128),  # decoder block #2
    (256, 128, 128),  # decoder block #3
    (512, 256, 256),  # decoder block #4
    (512, 1024, 512)  # decoder block #5
)

# large models
CONV_CHANNELS_L = (
    (128, 32, 32, 128),      # output residual block
    (128, 32, 32, 128),      # decoder block #1
    (512, 128, 128, 512),    # decoder block #2
    (1024, 256, 256, 1024),  # decoder block #3
    (2048, 512, 512, 2048),  # decoder block #4
    (2048, 4096, 2048)       # decoder block #5
)


class UNet(nn.Module):
    """
    U-Net with a ResNet backbone.
    """
    def __init__(self, resnet_layers: int = 34, cie_lab: bool = False,
                 blur: bool = False, blur_final: bool = True,
                 self_attention: bool = False, res_blocks: bool = False,
                 **kwargs):
        super().__init__()
        assert 'self_attention' not in kwargs

        # select and download pretrained ResNet
        if resnet_layers == 18:
            resnet = models.resnet18(pretrained=True)
        elif resnet_layers == 34:
            resnet = models.resnet34(pretrained=True)
        elif resnet_layers == 50:
            resnet = models.resnet50(pretrained=True)
        elif resnet_layers == 101:
            resnet = models.resnet101(pretrained=True)
        elif resnet_layers == 152:
            resnet = models.resnet152(pretrained=True)
        else:
            raise ValueError('Invalid number of ResNet layers ' +
                             f'({resnet_layers})')
        large = resnet_layers > 34
        c_conv = CONV_CHANNELS_L if large else CONV_CHANNELS_S

        # numbers of input and output channels
        inp_dim = 3                    # repeated grayscale or L*
        out_dim = 2 if cie_lab else 3  # RGB or a*b*

        # encoder
        # comments show output shape given (3, 224, 224) input
        self.enc_block1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu
                                        )  # -> (64, 112, 112)
        self.enc_block2 = nn.Sequential(resnet.maxpool, resnet.layer1
                                        )  # -> (64 | 256, 56, 56)
        self.enc_block3 = resnet.layer2    # -> (128 | 512, 28, 28)
        self.enc_block4 = resnet.layer3    # -> (256 | 1024, 14, 14)
        self.enc_block5 = resnet.layer4    # -> (512 | 2048, 7, 7)

        # choose convolution blocks to use in decoder
        if res_blocks:
            self.ConvBlock33 = layers.ResidualUnit33
            self.ConvBlock131 = layers.ResidualUnit131
        else:
            self.ConvBlock33 = layers.ConvBlock33
            self.ConvBlock131 = layers.ConvBlock131

        # decoder
        # c_conv -- numbers of channels in convolutional blocks
        # c_out -- output channels (after upscaling)
        # c_out = |next_input| - |skip_connection|
        self.dec_block5 = self._dec_block(
            c_conv=c_conv[5],
            c_out=c_conv[4][0] - (1024 if large else 256),
            blur=blur,
            **kwargs
        )
        self.dec_block4 = self._dec_block(
            c_conv=c_conv[4],
            c_out=c_conv[3][0] - (512 if large else 128),
            blur=blur,
            **kwargs
        )
        self.dec_block3 = self._dec_block(
            c_conv=c_conv[3],
            c_out=c_conv[2][0] - (256 if large else 64),
            blur=blur,
            self_attention=self_attention,
            **kwargs
        )
        self.dec_block2 = self._dec_block(
            c_conv=c_conv[2],
            c_out=c_conv[1][0] - 64,
            blur=blur,
            **kwargs
        )
        self.dec_block1 = self._dec_block(
            c_conv=c_conv[1],
            c_out=c_conv[0][0],
            blur=(blur and blur_final),
            **kwargs
        )

        # final convolution
        if len(c_conv[0]) == 3:
            res_block = layers.ResidualUnit33
        else:
            res_block = layers.ResidualUnit131
        self.output_conv = nn.Sequential(
            res_block([c + inp_dim for c in c_conv[0]], **kwargs),
            nn.Conv2d(c_conv[0][-1] + inp_dim, out_dim, 1, 1, 0),
            nn.Sigmoid()
        )

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
    def _dec_block(self, c_conv: Sequence[int], c_out: int, blur: bool,
                   self_attention: bool = False, **kwargs) -> nn.Module:
        if len(c_conv) == 3:
            conv = self.ConvBlock33
        else:
            assert len(c_conv) == 4
            conv = self.ConvBlock131
        return nn.Sequential(
            conv(c_conv, self_attention=self_attention, **kwargs),
            layers.PixelShuffle_ICNR(c_conv[-1], c_out*4, scale=2, blur=blur),
        )
