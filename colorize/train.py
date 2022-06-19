from typing import Union, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset_utils import ColorizationFolderDataset, tensor2image
from torchvision import models, transforms as T
from generators import UNet
# from loss_functions import VGG16Loss


BATCH_SIZE = 16
BATCHES_PER_UPDATE = 1
OUTPUT_DIR = 'output'
UPDATES_PER_EVAL = 500
TOTAL_UPDATES = 500 * 20
EXPORT_IMAGES = 64
LR = 1e-4
WEIGHT_DECAY = 0
GRAD_CLIP = None
CIE_LAB = True


def main():
    # load dataset
    train_dataset = ColorizationFolderDataset(
        folder='data/small/train',
        transforms=T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
        ]),
        cie_lab=CIE_LAB
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataset = ColorizationFolderDataset(
        folder='data/small/val',
        cie_lab=CIE_LAB
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False)

    # initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        resnet=models.resnet18(pretrained=True),
        bn=False,
        kaiming_init=True,
        cie_lab=CIE_LAB
    ).to(device)
    model.freeze_encoder()
    # loss_fn = VGG16Loss(
    #     feat_layers=['relu3_3', 'relu4_3', 'relu5_3'],
    #     feat_weights=[0.3, 1.0, 0.15],
    #     style_layers=[],
    #     style_weights=[],
    #     base_loss=nn.L1Loss
    # ).to(device)
    loss_fn = nn.L1Loss()

    # export first batch of validation set images
    if EXPORT_IMAGES:
        exported = 0
        for X, _, Y in val_dataloader:
            for i in range(min(EXPORT_IMAGES - exported, len(Y))):
                img = tensor2image(Y[i], X[i], cie_lab=CIE_LAB)
                img.save(os.path.join(OUTPUT_DIR, f'{exported}_real.jpeg'))
                exported += 1
            if exported == EXPORT_IMAGES:
                break

    # train the model
    train_losses, val_losses = train(
        model, train_dataloader, val_dataloader, loss_fn, device,
        total_iterations=TOTAL_UPDATES, eval_every=UPDATES_PER_EVAL,
        export_images=EXPORT_IMAGES)

    # plot losses
    updates = np.arange(1, len(train_losses) + 1) * UPDATES_PER_EVAL
    plt.figure()
    plt.plot(updates, train_losses, label='train')
    plt.plot(updates, val_losses, label='validate')
    plt.xlabel('Parameter updates')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss.png'))


def train(model: nn.Module, train_dataloader: DataLoader,
          val_dataloader: nn.Module, loss_fn: nn.Module, device: torch.device,
          eval_every: int = 100, total_iterations: int = 50000,
          export_images: Optional[int] = None, save_best: bool = True
          ) -> np.ndarray:
    num_updates = 0
    cur_iter = 0
    cur_loss = 0.0
    cur_samples = 0
    train_loss = []
    val_loss = []
    optimizer = optim.Adam(params=model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=3, verbose=True)
    model.train()

    while True:
        # forward pass
        for _, Xn, Y in train_dataloader:
            Xn, Y = Xn.to(device), Y.to(device)
            output = model(Xn)
            loss = loss_fn(output, Y)
            assert not torch.isnan(loss).any()

            # update average loss for current `eval_every` iterations
            cur_loss += loss.item() * len(Y)  # weighted average
            cur_samples += len(Y)

            # backprop and parameter update
            loss.backward()
            cur_iter += 1
            if cur_iter == BATCHES_PER_UPDATE:
                if GRAD_CLIP:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                cur_iter = 0
                num_updates += 1
                print(f'\r[{num_updates} iterations]', end='')

                # validation
                if num_updates % eval_every == 0 and num_updates != 0:
                    # calculate and save current train loss
                    train_loss.append(cur_loss / cur_samples)
                    print(f'\n  train loss: {train_loss[-1]:.2e}')
                    cur_loss = 0.0
                    cur_samples = 0

                    # evaluate
                    val_loss.append(
                        evaluate(model, val_dataloader, loss_fn, device,
                                 export_images, num_updates)
                    )
                    print(f'  val loss: {val_loss[-1]:.2e}')
                    scheduler.step(val_loss[-1])
                    if save_best and np.argmin(val_loss) == len(val_loss)-1:
                        torch.save(model.state_dict(),
                                   os.path.join(OUTPUT_DIR, 'model.pth'))
                    model.train()

            if num_updates >= total_iterations:
                return np.array(train_loss), np.array(val_loss)


@torch.no_grad()
def evaluate(model: nn.Module, val_dataloader: DataLoader, loss_fn: nn.Module,
             device: torch.device, export_images: Optional[int] = None,
             num_iter: Union[int, str] = '') -> float:
    model.eval()
    loss = 0.0
    samples = 0
    to_export = 0 if export_images is None else export_images

    for X, Xn, Y in val_dataloader:
        Xn, Y = Xn.to(device), Y.to(device)
        output = model(Xn)
        loss += loss_fn(output, Y).item() * len(Y)
        samples += len(Y)

        if to_export:  # save generated images
            for i in range(min(len(output), to_export)):
                img = tensor2image(output[i], L=X[i], cie_lab=CIE_LAB)
                fname = f'{export_images - to_export}_{num_iter}.jpeg'
                img.save(os.path.join(OUTPUT_DIR, fname))
                to_export -= 1

    return loss / samples


if __name__ == '__main__':
    main()
