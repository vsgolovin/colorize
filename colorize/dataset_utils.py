from typing import Optional, Callable
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision as tv

CIC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CIC_DIR / 'colorization'))
sys.path.append(str(CIC_DIR))

from colorizers import *


NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


class ColorizationFolderDataset(Dataset):
    """
    Read dataset from a folder of images.
    """

    def __init__(
        self,
        folder: str,
        transforms: Optional[Callable] = None,
        image_format: str = "JPEG",
    ):
        assert os.path.exists(folder)
        self.folder = folder
        self.files = tuple(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(image_format)
        )
        self.transforms = transforms
        self.grayscale = tv.transforms.Grayscale(num_output_channels=3)
        self.to_tensor = tv.transforms.ToTensor()
        self.normalize = tv.transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        with Image.open(self.files[index]) as img:
            if self.transforms:
                img = self.transforms(img)
            gray = self.to_tensor(self.grayscale(img))
            gray = self.normalize(gray)
            img = self.to_tensor(img)
        return gray, img

    def __len__(self):
        return len(self.files)


class ColorizationDataset(Dataset):
    def __init__(self, data: Dataset, transforms: Optional[Callable] = None):
        self.original_dataset = data  # some image dataset
        self.grayscale = tv.transforms.Grayscale(num_output_channels=3)
        self.to_tensor = tv.transforms.ToTensor()
        self.normalize = tv.transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        img = self.original_dataset.__getitem__(index)[0]
        if self.transforms:
            img = self.transforms(img)
        gray = self.to_tensor(self.grayscale(img))
        gray = self.normalize(gray)
        img = self.to_tensor(img)
        return gray, img

    def __len__(self):
        return len(self.original_dataset)


class CICFolderDataset(Dataset):
    """
    Read dataset from a folder of images.
    """

    def __init__(
        self,
        folder: str,
        transforms: Optional[Callable] = None,
        image_format: str = "JPEG",
    ):
        assert os.path.exists(folder)
        self.folder = folder
        self.files = tuple(f
            for f in os.listdir(folder)
            if f.endswith(image_format)
        )
        #self.grayscale = tv.transforms.Grayscale(num_output_channels=3)
        self.to_tensor = tv.transforms.ToTensor()

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        with Image.open(Path(self.folder, self.files[index])) as img:
            #gray = np.asarray(self.grayscale(img))
            gray = np.asarray(img)
            tens_l_orig, tens_l_rs = preprocess_img(gray, HW=(256, 256))
            img = self.to_tensor(img)
        return {'img_name': self.files[index],
                'tens_l_orig': tens_l_orig[0], 
                'tens_l_rs': tens_l_rs[0], 
                'img': img}

    def __len__(self):
        return len(self.files)

class MetrixDataset(Dataset):
    """
    Read dataset from original and colorized folder.
    """

    def __init__(
        self,
        orig_folder: str,
        colorized_folder: str,
        image_format: str = "JPEG",
    ):
        assert set(os.listdir(colorized_folder)).issubset(set(os.listdir(orig_folder)))
        assert os.path.exists(orig_folder)
        assert os.path.exists(colorized_folder)
        
        self.orig_folder = orig_folder
        self.colorized_folder = colorized_folder
        self.filenames = tuple(f
            for f in os.listdir(colorized_folder)
            if f.endswith(image_format)
        )
        self.to_tensor = tv.transforms.ToTensor()

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        with Image.open(Path(self.orig_folder, self.filenames[index])) as orig_image:
            orig_image = self.to_tensor(orig_image)
        with Image.open(Path(self.colorized_folder, self.filenames[index])) as colorized_image:
            colorized_image = self.to_tensor(colorized_image)
        return {'orig_image': orig_image,
                'colorized_image': colorized_image}

    def __len__(self):
        return len(self.filenames)


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


def tensor2image(x: torch.tensor) -> Image:
    x = x.cpu().detach()
    x = torch.clamp(x, 0, 1)
    return tv.transforms.ToPILImage()(x)
