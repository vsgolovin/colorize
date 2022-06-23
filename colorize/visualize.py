import os
from PIL import Image
import torch
from colorize.utils import ColorizationFolderDataset, tensor2image, rescale4resnet
from colorize.generators import UNet


OUTPUT_DIR = 'output'
CIE_LAB = True


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
        resnet_layers=34,
        self_attention=True,
        spectral_norm=True,
        blur=True,
        cie_lab=CIE_LAB
    ).to(device).eval()
    model.load_state_dict(torch.load('output/model.pth'))

    # colorize images one-by-one
    for i, [X, Xn, _] in enumerate(test_images):
        inp = Xn.to(device).unsqueeze(0)
        out = model(inp).squeeze()
        colored = tensor2image(out, X, cie_lab=CIE_LAB)
        colored.save(os.path.join(OUTPUT_DIR, f'{i}_colored.jpeg'))


if __name__ == '__main__':
    main()
