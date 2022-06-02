import torch
from dataset_utils import ColorizationDataset
import torchvision as tv
from generators import UNet


# initialize dataset
dataset = ColorizationDataset(
    data=tv.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True),
    transform=tv.transforms.Pad(96)  # resnet needs 224x224
)
model = UNet()

# get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# x - grayscale image tensor, y -- original image tensor
x, y = dataset[torch.randint(high=len(dataset), size=(1,))]
x, y = x.to(device).unsqueeze(0), y.to(device).unsqueeze(0)
output = model(x)

print(f'Input shape: {x.shape}')
print(f'Target shape: {y.shape}')
print(f'Output shape: {output.shape}')
