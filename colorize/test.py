import torch
from dataset_utils import ColorizationFolderDataset, tensor2image
import torchvision as tv
from generators import UNet


# initialize dataset
dataset = ColorizationFolderDataset(
    folder='data/val',
    transforms=tv.transforms.RandomCrop(224)
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

original = tensor2image(y.squeeze())
original.show()
output = tensor2image(output.squeeze())
output.show()
