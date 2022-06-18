import os
from PIL import Image
import torch
from dataset_utils import ColorizationFolderDataset, tensor2image
from generators import UNet34


OUTPUT_DIR = 'output'


@torch.no_grad()
def main():
    # load images and pretrained model
    test_images = ColorizationFolderDataset(
        'data/tiny/test',
        transforms=rescale4resnet
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet34().to(device).eval()
    model.load_state_dict(torch.load('output/model.pth'))

    # colorize images one-by-one
    for i, [image, _] in enumerate(test_images):
        inp = image.to(device).unsqueeze(0)
        out = model(inp).squeeze()
        colored = tensor2image(out)
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
