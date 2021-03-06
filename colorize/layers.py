from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ResidualUnitBase(nn.Module):
    """
    Base class for residual units.
    See "Identity Mappings in Deep Residual Networks" (arXiv:1603.05027).
    """
    def __init__(self, channels: tuple[int], bn: bool = True,
                 bn_first: bool = True, pre_activation: bool = False,
                 identity_conv: bool = False):
        super().__init__()
        self.use_bn = bn
        self.bn_first = bn_first
        self.pre_activation = pre_activation
        self.residual_function = lambda _: 0  # replace in child classes
        self.channels = channels
        if not identity_conv == 'slice':
            self.identity = self._identity_slice
        else:
            self.identity = nn.Conv2d(channels[0], channels[-1], 1, 1, 0)

    def _identity_slice(self, x: Tensor) -> Tensor:
        c_in, c_out = self.channels[0], self.channels[-1]
        if c_in == c_out:
            return x
        elif c_in > c_out:
            return x[..., :c_out, :, :]
        else:
            raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        output = self.identity(x) + self.residual_function(x)
        if self.pre_activation:
            return output
        return F.relu(output)

    # constructor methods
    def _activation(self, num_channels: int) -> nn.Module:
        if not self.use_bn:
            return nn.ReLU()
        if self.bn_first:
            return nn.Sequential(self._batchnorm(num_channels), nn.ReLU())
        return nn.Sequential(nn.ReLU(), self._batchnorm(num_channels))

    @staticmethod
    def _batchnorm(num_channels: int) -> nn.Module:
        return nn.BatchNorm2d(num_channels)

    # weight initialization
    def _initialize(self):
        # copied from torchvision resnet implementation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _apply_spectral_norm(layers: list[nn.Module]):
        return [layer if not isinstance(layer, nn.Conv2d)
                else nn.utils.spectral_norm(layer)
                for layer in layers]


class ResidualUnit33(ResidualUnitBase):
    """
    Residual block with 3x3 -> 3x3 convolutions.
    """
    def __init__(self, channels: tuple[int, int, int], bn: bool = True,
                 bn_first: bool = True, pre_activation: bool = False,
                 identity_conv: bool = False, kaiming_init: bool = True,
                 conv_bias: bool = False, self_attention: bool = False,
                 spectral_norm: bool = False):
        super().__init__(channels, bn, bn_first, pre_activation, identity_conv)

        # residual function
        rf_modules = [nn.Conv2d(channels[0], channels[1], 3, 1, 1,
                                bias=conv_bias),
                      self._activation(channels[1]),
                      nn.Conv2d(channels[1], channels[2], 3, 1, 1,
                                bias=conv_bias)]
        if spectral_norm:
            rf_modules = self._apply_spectral_norm(rf_modules)
        if self_attention:
            rf_modules.insert(-1, SelfAttention(channels[1]))
        if pre_activation:
            rf_modules = [self._activation(channels[0])] + rf_modules
        elif self.use_bn:
            rf_modules = rf_modules + [self._batchnorm(channels[2])]
        self.residual_function = nn.Sequential(*rf_modules)

        # weight initialization
        if kaiming_init:
            self._initialize()


class ResidualUnit131(ResidualUnitBase):
    """
    Residual block with 1x1 -> 3x3 -> 1x1 convolutions.
    """
    def __init__(self, channels: tuple[int, int, int, int], bn: bool = True,
                 bn_first: bool = True, pre_activation: bool = False,
                 identity_conv: bool = False, kaiming_init: bool = True,
                 conv_bias: bool = False, self_attention: bool = False,
                 spectral_norm: bool = False):
        super().__init__(channels, bn, bn_first, pre_activation, identity_conv)

        # residual function
        rf_modules = [nn.Conv2d(channels[0], channels[1], 1, 1, 0,
                                bias=conv_bias),
                      self._activation(channels[1]),
                      nn.Conv2d(channels[1], channels[2], 3, 1, 1,
                                bias=conv_bias),
                      self._activation(channels[2]),
                      nn.Conv2d(channels[2], channels[3], 1, 1, 0,
                                bias=conv_bias)]
        if spectral_norm:
            rf_modules = self._apply_spectral_norm(rf_modules)
        if self_attention:
            rf_modules.insert(-1, SelfAttention(channels[2]))
        if pre_activation:
            rf_modules = [self._activation(channels[0])] + rf_modules
        elif self.use_bn:
            rf_modules = rf_modules + [self._batchnorm(channels[3])]
        self.residual_function = nn.Sequential(*rf_modules)

        # weight initialization
        if kaiming_init:
            self._initialize()


class ConvBlock33(ResidualUnit33):
    def __init__(self, channels: tuple[int, int, int], bn: bool = True,
                 bn_first: bool = True, kaiming_init: bool = False,
                 conv_bias: bool = True, self_attention: bool = False,
                 spectral_norm: bool = False):
        super().__init__(channels, bn, bn_first, pre_activation=False,
                         identity_conv=False, kaiming_init=kaiming_init,
                         conv_bias=conv_bias, self_attention=self_attention,
                         spectral_norm=spectral_norm)
        self.identity = lambda _: 0


class ConvBlock131(ResidualUnit131):
    def __init__(self, channels: tuple[int, int, int], bn: bool = True,
                 bn_first: bool = True, kaiming_init: bool = False,
                 conv_bias: bool = True, self_attention: bool = False,
                 spectral_norm: bool = False):
        super().__init__(channels, bn, bn_first, pre_activation=False,
                         identity_conv=False, kaiming_init=kaiming_init,
                         conv_bias=conv_bias, self_attention=self_attention,
                         spectral_norm=spectral_norm)
        self.identity = lambda _: 0


class SelfAttention(nn.Module):
    """
    Self-attention module proposed in the SAGAN paper (arXiv:1805.08318).
    """
    def __init__(self, n_channels: int, qk_downscale: int = 8):
        assert qk_downscale >= 1
        super().__init__()
        self.Q = self._conv(n_channels, n_channels // qk_downscale)
        self.K = self._conv(n_channels, n_channels // qk_downscale)
        self.V = self._conv(n_channels, n_channels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _conv(c_in: int, c_out: int) -> nn.Module:
        return nn.utils.spectral_norm(
            nn.Conv1d(c_in, c_out, 1, 1, 0, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        size = x.size()
        x = x.flatten(start_dim=2)
        q, k, v = self.Q(x), self.K(x), self.V(x)
        qk = self.softmax(torch.bmm(q.transpose(1, 2), k))
        out = self.gamma * torch.bmm(v, qk) + x
        return out.reshape(size)


def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni, nf, h, w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)


class PixelShuffle_ICNR(nn.Module):
    """
    PixelShuffle upsampling with ICNR weight initialization
    """
    def __init__(self, c_in: int, c_out: Optional[int] = None, scale: int = 2,
                 blur: bool = False):
        super().__init__()
        c_out = c_out if c_out is not None else c_in * scale ** 2
        conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        conv.weight.data.copy_(icnr_init(conv.weight.data))
        self.conv = conv
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.shuffle = nn.PixelShuffle(scale)
        if blur:
            self.blur = nn.Sequential(
                nn.ReflectionPad2d((1, 0, 1, 0)),
                nn.AvgPool2d((2, 2), stride=1)
            )
        else:
            self.blur = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.shuffle(x)
        return self.blur(x) if self.blur else x
