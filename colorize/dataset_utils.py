from typing import Optional, Callable
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision as tv


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
        self.files = tuple(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(image_format)
        )
        self.grayscale = tv.transforms.Grayscale(num_output_channels=3)
        self.to_tensor = tv.transforms.ToTensor()

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        with Image.open(self.files[index]) as img:
            gray = np.asarray(self.grayscale(img))
            tens_l_orig, tens_l_rs = preprocess_img(gray, HW=(256, 256))
            img = self.to_tensor(img)
        return tens_l_orig[0], tens_l_rs[0], img

    def __len__(self):
        return len(self.files)


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
