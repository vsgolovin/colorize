from typing import Union, Optional, Callable
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision as tv
from loss_functions import VGG16Loss
from dataset_utils import tensor2image, CICFolderDataset
from tqdm import tqdm

CIC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CIC_DIR / 'colorization'))
sys.path.append(str(CIC_DIR))

from colorizers import *

BATCH_SIZE = 32
OUTPUT_DIR = 'output'
EXPORT_IMAGES = True

def main():
    # load dataset
    val_dataset = CICFolderDataset(
        folder='../data/val'
    )  # no cropping, already 224x224
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False)

    # initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    colorizer_eccv16 = eccv16(pretrained=True).to(device)
    colorizer_siggraph17 = siggraph17(pretrained=True).to(device)
    loss_fn = VGG16Loss(
        feat_layers=['relu2_2', 'relu3_2', 'relu4_2'],
        feat_weights=[1., 0.5, 0.5],
        style_layers=[],
        style_weights=[],
        base_loss=nn.MSELoss
    ).to(device)

    eccv16_loss = CIC_evaluate(colorizer_eccv16, 'ECCV16', val_dataloader, loss_fn, device, EXPORT_IMAGES)
    siggraph17_loss = CIC_evaluate(colorizer_siggraph17, 'siggraph17', val_dataloader, loss_fn, device, EXPORT_IMAGES)

    print(f'eccv16_VGG16_loss: {eccv16_loss}')
    print(f'siggraph17_VGG16_loss: {siggraph17_loss}')

    
@torch.no_grad()
def CIC_evaluate(model: Union[ECCVGenerator, SIGGRAPHGenerator], model_name: str, val_dataloader: DataLoader, loss_fn: nn.Module, device: torch.device, export_first: bool = False):
    model.eval()
    loss = 0.0
    samples = 0
    to_export = export_first

    for X, Y, Z in tqdm(val_dataloader, desc=model_name):
        Y, Z = Y.to(device), Z.to(device)
        result = model(Y).cpu()
        output = np.array([postprocess_tens(a, b) for a, b in list(zip(X.unsqueeze(1), result.unsqueeze(1)))])
        output = torch.tensor(output).swapaxes(1, 3).transpose(2, 3).to(device)
        loss += loss_fn(output, Z).item() * len(Z)
        samples += len(Y)

        if to_export:  # export images generated in the first batch
                for i, t in enumerate(output):
                    img = tensor2image(t)
                    fname = os.path.join(OUTPUT_DIR, f'{i}_{model_name}.jpeg')
                    img.save(fname)
                for i, t in enumerate(Z):
                    img = tensor2image(t)
                    fname = os.path.join(OUTPUT_DIR, f'{i}_original.jpeg')
                    img.save(fname)
                to_export = False

    return loss / samples

if __name__ == '__main__':
    main()

