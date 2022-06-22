from typing import Optional
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from generators import UNet
from critics import SimpleCritic
from dataset_utils import ColorizationFolderDataset
from gan_learner import GANLearner


BATCH_SIZE = 64
MODEL_PATH = 'checkpoints/resnet18.pth'
TRAIN_DATA = 'data/imagenet_tiny/train'
VAL_DATA = 'data/imagenet_tiny/val'
OUTPUT_FOLDER = 'output'
CIE_LAB = True  # RGB not yet supported
EVAL_EVERY = 10
TOTAL_ITERATIONS = EVAL_EVERY * 10


def main():
    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load pretrained generators
    net_G = UNet(
            resnet_layers=18,
            cie_lab=CIE_LAB
        ).eval().to(device)
    net_G.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # initialize discriminator
    net_D = SimpleCritic()

    # load dataset
    train_dataset = ColorizationFolderDataset(
            folder=TRAIN_DATA,
            transforms=T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
            ]),
            cie_lab=CIE_LAB
        )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # train the model
    learner = GANLearner(net_G, net_D, device='cuda')
    learner.freeze_generator()
    train_loss, train_acc = train(
        learner,
        train_dataloader,
        None,
        eval_every=EVAL_EVERY,
        total_iterations=TOTAL_ITERATIONS
    )
    torch.save(learner.net_D.state_dict(),
               os.path.join(OUTPUT_FOLDER, 'model.pth'))

    # plot loss curve
    iters = np.arange(1, len(train_loss) + 1) * EVAL_EVERY
    plt.figure()
    plt.plot(iters, train_loss, label='train')
    plt.xlabel('Parameter updates')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(iters, train_acc, label='train')
    plt.xlabel('Parameter updates')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def train(learner: GANLearner, train_dataloader: DataLoader,
          val_dataloader: DataLoader, eval_every: int = 100,
          total_iterations: int = 1000) -> tuple[np.ndarray]:

    cur_iter = 0
    cur_loss = 0.0
    cur_samples = 0
    accurate_predictions = 0
    train_loss = []
    train_acc = []
    # val_loss = []
    # val_acc = []
    pbar = None

    while True:
        for batch in train_dataloader:
            # create progress bar (if needed)
            if pbar is None:
                pbar = tqdm(total=eval_every)

            # single discriminator iteration
            bs = len(batch[0]) * 2  # current batch size
            lossD, _, acc = learner.train_iter(batch, True)

            # update metrics
            cur_loss += lossD * bs
            accurate_predictions += acc
            cur_samples += bs
            pbar.update(1)
            cur_iter += 1

            # actions for current 'eval every'
            if cur_iter % eval_every == 0:
                pbar.close()
                pbar = None
                train_loss.append(cur_loss / cur_samples)
                train_acc.append(accurate_predictions / cur_samples)
                print(f'Discriminator train loss: {train_loss[-1]:.2e}'
                      + f' accuracy: {train_acc[-1]:.3f}')
                cur_loss = 0.0
                accurate_predictions = 0
                cur_samples = 0

            if cur_iter >= total_iterations:
                if pbar is not None:
                    pbar.close()
                return np.array(train_loss), np.array(train_acc)


if __name__ == '__main__':
    main()
