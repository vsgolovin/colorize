from typing import Optional, Callable
import os
from pathlib import Path
import warnings
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision as tv
import numpy as np
from skimage.color import rgb2lab, lab2rgb


NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


class ColorizationFolderDataset(Dataset):
    """
    Read dataset from a folder of images.
    """
    def __init__(self, folder: str, transforms: Optional[Callable] = None,
                 image_format: str = 'JPEG', cie_lab: bool = False):
        assert os.path.exists(folder)
        self.folder = folder
        self.files = tuple(os.path.join(folder, f) for f in os.listdir(folder)
                           if f.endswith(image_format))
        self.transforms = transforms
        self.grayscale = tv.transforms.Grayscale(num_output_channels=1)
        self.to_tensor = tv.transforms.ToTensor()
        self.normalize = tv.transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        self.use_cielab = cie_lab

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        with Image.open(self.files[index]) as img:
            if self.transforms:
                img = self.transforms(img)
            if self.use_cielab:
                img = np.array(img)
                if img.ndim == 2:
                    img = img[:, :, np.newaxis]
                    img = np.broadcast_to(img[:, :, np.newaxis],
                                          (*img.shape[:2], 3))
                arr = rgb2lab(img).astype('float32')
                L, ab = map(self.to_tensor, (arr[:, :, 0], arr[:, :, 1:]))
                gray = L / 100.
                target = (ab + 128.) / 255.
            else:
                gray = self.to_tensor(self.grayscale(img))
                target = self.to_tensor(img)
        return gray, self.normalize(gray.repeat((3, 1, 1))), target#, self.files[index]

    def __len__(self):
        return len(self.files)


class LabColorizationDataset(Dataset):
    def __init__(self,
                 real_image_folder: str,
                 fake_image_folder: str,
                 transforms: Optional[Callable] = None,
                 image_format: str = "JPEG"):
        assert set(os.listdir(real_image_folder)) ==\
               set(os.listdir(fake_image_folder))
        self.real_folder = real_image_folder
        self.fake_folder = fake_image_folder
        self.filenames = tuple(
            f for f in os.listdir(real_image_folder)
            if f.endswith(image_format))
        self.to_tensor = tv.transforms.ToTensor()
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        with Image.open(
                Path(self.real_folder, self.filenames[index])) as real_img:
            real_img = self.transforms(real_img)
            real_L, real_ab = self.lab_transform(real_img)
        with Image.open(
                Path(self.fake_folder, self.filenames[index])) as fake_img:
            fake_img = self.transforms(fake_img)
            fake_L, fake_ab = self.lab_transform(fake_img)
        return {'real_L': real_L, 'real_ab': real_ab,
                'fake_L': fake_L, 'fake_ab': fake_ab}

    def lab_transform(self, img):
        if self.transforms:
            img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = self.to_tensor(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
        return L, ab

    def __len__(self):
        return len(self.filenames)


class ColorizationDataset(Dataset):
    def __init__(self, data: Dataset, transforms: Optional[Callable] = None):
        self.original_dataset = data  # some image dataset
        self.grayscale = tv.transforms.Grayscale(num_output_channels=1)
        self.to_tensor = tv.transforms.ToTensor()
        self.normalize = tv.transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        self.transforms = transforms

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        img = self.original_dataset.__getitem__(index)[0]
        if self.transforms:
            img = self.transforms(img)
        gray = self.to_tensor(self.grayscale(img))
        target = self.to_tensor(img)
        return gray, self.normalize(gray.repeat((3, 1, 1))), target

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


def tensor2image(x: Tensor, L: Optional[Tensor] = None,
                 cie_lab: bool = False) -> Image:
    x = x.cpu().detach()
    x = torch.clamp(x, 0, 1)
    if cie_lab:
        L = torch.clamp(L.cpu().detach(), 0, 1)
        L = L * 100
        ab = x * 255. - 128.
        lab = torch.cat([L, ab], dim=0).permute(1, 2, 0).numpy()
        with warnings.catch_warnings():  # not all a*b* pairs are valid
            warnings.simplefilter('ignore')
            x = lab2rgb(lab)
        x = np.uint8(np.round(x * 255.))
    return tv.transforms.ToPILImage()(x)


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
