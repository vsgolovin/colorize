from torch import nn, Tensor
from layers import SelfAttention


class DeOldify_Discriminator(nn.Module):
    def __init__(self, c_in: int = 3, n_filters: int = 256, n_down: int = 3):
        super().__init__()
        # First convolution
        model = [nn.Conv2d(c_in, n_filters, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                 nn.Dropout2d(p=0.075, inplace=False)]
        # Blocks like in DeOldify
        for i in range(n_down):
            model += self.get_block(n_filters, self_attention=(i == 0))
            n_filters *= 2
        # Final convolution
        model += [
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(n_filters, 1, kernel_size=4, stride=1, bias=False)
        ]
        self.model = nn.Sequential(*model)

    def get_block(self, c_in, self_attention=False, dropout=0.15):
        layers = [
            nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=dropout, inplace=False),
            nn.Conv2d(c_in, c_in * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]
        if self_attention:
            layers += [SelfAttention(c_in * 2)]
        return layers

    def forward(self, x):
        return self.model(x)


class SimpleCritic(nn.Module):
    def __init__(self, in_channels: int = 3, nc_first: int = 64,
                 num_blocks: int = 3):
        super().__init__()
        # first downscale block
        self.inp_block = self._conv_block(in_channels, nc_first)
        nc = nc_first  # current number of channels

        # next `num_blocks` downscale blocks
        modules = []
        for _ in range(num_blocks):
            modules.append(self._conv_block(nc, nc * 2))
            nc *= 2
        self.downscale = nn.Sequential(*modules)

        # output block
        self.out = nn.Sequential(
            nn.Conv2d(nc, 1, 4, 1, bias=False),
            nn.Sigmoid(),
            nn.Flatten(start_dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.inp_block(x)
        x = self.downscale(x)
        return self.out(x).mean(1)  # returns probabilities

    @staticmethod
    def _conv_block(c_in: int, c_out: int):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1)
        )
