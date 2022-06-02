from typing import Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ColorizationDataset(Dataset):
    def __init__(self, data: Dataset, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.original_dataset = data  # some image dataset
        self.grayscale = transforms.Grayscale(num_output_channels=3)
        self.trasform_out = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # optional input (grayscale) and output image transforms
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        img = self.original_dataset.__getitem__(index)[0]
        gray = self.grayscale(img)
        if self.transform:
            gray = self.transform(gray)
        if self.target_transform:
            img = self.target_transform(img)
        return self.trasform_out(gray), self.trasform_out(img)

    def __len__(self):
        return len(self.original_dataset)


def crop_resize(img: Image, new_width: int, new_height: int) -> Image:
    kx, ky = img.width / new_width, img.height / new_height
    if kx > ky:  # crop horizontally
        w = int(round(new_width * ky))
        left = (img.width - w) // 2
        img = img.crop((left, 0, left + w, img.height))
    else:  # crop vertically
        h = int(round(new_height * kx))
        upper = (img.height - h) // 2
        img = img.crop((0, upper, img.width, upper + h))
    return img.resize((new_width, new_height))
