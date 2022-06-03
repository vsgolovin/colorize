from typing import Union
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_utils import ColorizationFolderDataset, tensor2image
import torchvision as tv
from generators import UNet
from loss_functions import VGG16Loss


BATCH_SIZE = 32
OUTPUT_DIR = 'output'
UPDATES_PER_EVAL = 100
TOTAL_UPDATES = 50000
EXPORT_IMAGES = True


def main():
    # load dataset
    train_dataset = ColorizationFolderDataset(
        folder='data/train',
        transforms=tv.transforms.RandomCrop(224)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataset = ColorizationFolderDataset(
        folder='data/val'
    )  # no cropping, already 224x224
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False)

    # initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    loss_fn = VGG16Loss(
        feat_layers=['relu2_2', 'relu3_2', 'relu4_2'],
        feat_weights=[1., 0.5, 0.5],
        style_layers=[],
        style_weights=[],
        base_loss=nn.MSELoss
    ).to(device)

    # export first batch of validation set images
    if EXPORT_IMAGES:
        for _, Y in val_dataloader:
            for i, y in enumerate(Y):
                img = tensor2image(y)
                img.save(os.path.join(OUTPUT_DIR, f'{i}_real.jpeg'))
            break

    # train the model
    train_losses, val_losses = train(
        model, train_dataloader, val_dataloader, loss_fn, device,
        total_iterations=TOTAL_UPDATES, eval_every=UPDATES_PER_EVAL,
        export_first=EXPORT_IMAGES)

    # plot losses
    iterations = np.arange(1, len(train_losses) + 1) * UPDATES_PER_EVAL
    plt.figure()
    plt.plot(iterations, train_losses, label='train')
    plt.plot(iterations, val_losses, label='validate')
    plt.xlabel('Parameter updates')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss.png'))


def train(model: nn.Module, train_dataloader: DataLoader,
          val_dataloader: nn.Module, loss_fn: nn.Module, device: torch.device,
          eval_every: int = 100, total_iterations: int = 50000,
          export_first: bool = True, save_best: bool = True
          ) -> np.ndarray:
    num_iter = 0
    cur_loss = 0.0
    cur_samples = 0
    train_losses = []
    val_losses = []
    optimizer = torch.optim.Adam(params=model.parameters())
    model.train()

    while True:
        # forward pass and parameter update
        for X, Y in train_dataloader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            loss = loss_fn(output, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update average loss for current `eval_every` iterations
            cur_loss += loss.item() * len(Y)  # weighted average
            cur_samples += len(Y)
            num_iter += 1

            if num_iter % eval_every == 0:
                # calculate and save current train loss
                train_losses.append(cur_loss / cur_samples)
                cur_loss = 0.0
                cur_samples = 0

                val_losses.append(
                    evaluate(model, val_dataloader, loss_fn, device,
                             export_first, num_iter)
                )
                print(f'[{num_iter} iterations]')
                print(f'  train loss: {train_losses[-1]:.2e}')
                print(f'  val loss: {val_losses[-1]:.2e}')
                if save_best and np.argmin(val_losses) == len(val_losses) - 1:
                    torch.save(model.state_dict(),
                               os.path.join(OUTPUT_DIR, 'model.pth'))
                model.train()

            if num_iter >= total_iterations:
                return np.array(train_losses), np.array(val_losses)


@torch.no_grad()
def evaluate(model: nn.Module, val_dataloader: DataLoader, loss_fn: nn.Module,
             device: torch.device, export_first: bool = True,
             num_iter: Union[int, str] = '') -> float:
    model.eval()
    loss = 0.0
    samples = 0
    to_export = export_first

    for X, Y in val_dataloader:
        X, Y = X.to(device), Y.to(device)
        output = model(X)
        loss += loss_fn(output, Y).item() * len(Y)
        samples += len(Y)

        if to_export:  # export images generated in the first batch
            for i, t in enumerate(output):
                img = tensor2image(t)
                fname = os.path.join(OUTPUT_DIR, f'{i}_{num_iter}.jpeg')
                img.save(fname)
            to_export = False

    return loss / samples


if __name__ == '__main__':
    main()
