from typing import Optional, Callable
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
