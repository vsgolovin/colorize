from torch import nn
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
