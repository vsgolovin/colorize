from typing import Union, Optional, Callable
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision as tv
from loss_functions import VGG16Loss
import torch
from torch import nn
from dataset_utils import tensor2image, CICFolderDataset, MetrixDataset
from make_dataset import Grayscale_folder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PARENT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PARENT_DIR / 'colorization'))
sys.path.append(str(PARENT_DIR / 'DeOldify'))
sys.path.append(str(PARENT_DIR))

from colorizers import *
from deoldify.visualize import *

BATCH_SIZE = 32
BASELINES = ['ECCV16', 'siggraph17', 'DeOldify_Artist', 'DeOldify_Stable']

VAL_FOLDER = '../data/val'
VAL_GRAY_FOLDER = '../data/val_gray'

ECCV16_FOLDER = '../data/benchmarks/ECCV16'
SIGGRAPH17_FOLDER = '../data/benchmarks/SIGGRAPH17'
ARTIST_FOLDER = '../data/benchmarks/DeOldify Artist'
STABLE_FOLDER = '../data/benchmarks/DeOldify Stable'

COLORIZE_FOLDERS = False
CALCULATE_METRIX = True

def CIC_colorize_folder(inp_folder: str, out_folder: int, colorizer: Union[ECCVGenerator, SIGGRAPHGenerator], image_format: str='JPEG'):
    """
    Colorize 3 channel GRAY! folder
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert os.path.exists(inp_folder)
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    val_dataset = CICFolderDataset(
        folder=inp_folder
    )  # no cropping, already 224x224
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False)
    for batch in tqdm(val_dataloader):
        filenames = batch['img_name']
        tens_l_orig = batch['tens_l_orig'].to(device)
        tens_l_rs = batch['tens_l_rs']
        raw_output = colorizer(tens_l_rs).cpu()
        output = np.array([postprocess_tens(a, b) for a, b in list(zip(tens_l_orig.unsqueeze(1), raw_output.unsqueeze(1)))])
        output = torch.tensor(output).swapaxes(1, 3).transpose(2, 3)
        for i in range(len(output)):
            tv.utils.save_image(output[i], Path(out_folder, filenames[i]))

def DeOldify_colorize_folder(inp_folder: str, out_folder: int, colorizer: ModelImageVisualizer, render_factor: int=35, image_format: str='JPEG'):
    assert os.path.exists(inp_folder)
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    filenames = tuple(f 
    for f in os.listdir(inp_folder) 
    if f.endswith(image_format)
    )
    for i in tqdm(range(len(filenames))):
        grayscale_img_path = Path(inp_folder, filenames[i])
        colorized_image = colorizer.get_transformed_image(path=grayscale_img_path, render_factor=render_factor)
        colorized_image.save(Path(out_folder, filenames[i]))

@torch.no_grad()
def Calculate_Metrics(orig_folder: str, colorized_folder: str, metrics):
    loss = np.zeros(len(metrics))
    samples = 0
    metrix_dataset = MetrixDataset(orig_folder, colorized_folder)
    metrix_dataloader = DataLoader(metrix_dataset, batch_size=BATCH_SIZE, shuffle=False)
    for batch in tqdm(metrix_dataloader):
        orig = batch['orig_image']
        color = batch['colorized_image']
        for i in range(len(metrics)):
            loss[i] += metrics[i](orig, color).item() * len(orig)
        samples += len(orig)
    loss /= samples
    for i in range(len(metrics)):
        print(f'{colorized_folder} -- {type(metrics[i])}: {loss[i]}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg16_loss = VGG16Loss(
        feat_layers=['relu2_2', 'relu3_2', 'relu4_2'],
        feat_weights=[1., 0.5, 0.5],
        style_layers=[],
        style_weights=[],
        base_loss=nn.MSELoss
    ).to(device)
    mse_loss = nn.MSELoss()
    metrics = [vgg16_loss, mse_loss]

    if not os.path.exists(VAL_GRAY_FOLDER):
        Grayscale_folder(VAL_FOLDER, VAL_GRAY_FOLDER, n_channels=3, n_pictures=1000)

    if COLORIZE_FOLDERS:
        if 'ECCV16' in BASELINES:
            Path(ECCV16_FOLDER).mkdir(parents=True, exist_ok=True)
            colorizer = eccv16(pretrained=True).to(device)
            CIC_colorize_folder(VAL_GRAY_FOLDER, ECCV16_FOLDER, colorizer)

        if 'siggraph17' in BASELINES:
            Path(SIGGRAPH17_FOLDER).mkdir(parents=True, exist_ok=True)
            colorizer = siggraph17(pretrained=True).to(device)
            CIC_colorize_folder(VAL_GRAY_FOLDER, SIGGRAPH17_FOLDER, colorizer)

        if 'DeOldify_Artist' in BASELINES:
            colorizer = get_image_colorizer(root_folder=Path('../DeOldify'), artistic=True)
            DeOldify_colorize_folder(VAL_GRAY_FOLDER, ARTIST_FOLDER, colorizer)

        if 'DeOldify_Stable' in BASELINES:
            colorizer = get_image_colorizer(root_folder=Path('../DeOldify'), artistic=False)
            DeOldify_colorize_folder(VAL_GRAY_FOLDER, STABLE_FOLDER, colorizer)

    if CALCULATE_METRIX:
        if 'ECCV16' in BASELINES:
            Calculate_Metrics(VAL_FOLDER, ECCV16_FOLDER, metrics)

        if 'siggraph17' in BASELINES:
            Calculate_Metrics(VAL_FOLDER, SIGGRAPH17_FOLDER, metrics)

        if 'DeOldify_Artist' in BASELINES:
            Calculate_Metrics(VAL_FOLDER, ARTIST_FOLDER, metrics)

        if 'DeOldify_Stable' in BASELINES:
            Calculate_Metrics(VAL_FOLDER, STABLE_FOLDER, metrics)

if __name__ == '__main__':
    main()
    