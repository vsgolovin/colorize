import os
from PIL import Image
import torch
from torchvision import models
from dataset_utils import ColorizationFolderDataset, tensor2image
from generators import UNet


OUTPUT_DIR = 'output'
CIE_LAB = False


@torch.no_grad()
def main():
    # load images and pretrained model
    test_images = ColorizationFolderDataset(
        'data/old_photos',
        transforms=rescale4resnet,
        cie_lab=CIE_LAB
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        resnet=models.resnet34(pretrained=True),
        bn=True,
        self_attention=True,
        kaiming_init=True,
        cie_lab=CIE_LAB
    ).to(device).eval()
    model.load_state_dict(torch.load('output/model.pth'))

    # colorize images one-by-one
    for i, [X, Xn, _] in enumerate(test_images):
        inp = Xn.to(device).unsqueeze(0)
        out = model(inp).squeeze()
        colored = tensor2image(out, X, cie_lab=CIE_LAB)
        colored.save(os.path.join(OUTPUT_DIR, f'{i}_colored.jpeg'))


def rescale4resnet(img: Image) -> Image:
    """
    Center crop (supposedly) large image to prevent shape mismatches inside
    a U-Net with a ResNet backbone.
    """
    w, h = img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    if new_w == w and new_h == h:
        return img
    left = (w - new_w) // 2
    upper = (h - new_h) // 2
    return img.crop((left, upper, left + new_w, upper + new_h))


if __name__ == '__main__':
    main()
