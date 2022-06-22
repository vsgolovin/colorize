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
TRAIN_FOLDER = 'data/train'
OUTPUT_FOLDER = 'output'
CIE_LAB = True  # RGB not yet supported
EVAL_EVERY = 150
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
            folder=TRAIN_FOLDER,
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
    train_loss = train(learner, train_dataloader, eval_every=EVAL_EVERY,
                       total_iterations=TOTAL_ITERATIONS, disc_only=True)[0]
    torch.save(learner.net_D.state_dict(),
               os.path.join(OUTPUT_FOLDER, 'model.pth'))

    # plot loss curve
    iters = np.arange(1, len(train_loss) + 1) * EVAL_EVERY
    plt.figure()
    plt.plot(iters, train_loss, label='train')
    plt.xlabel('Parameter updates')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def train(learner: GANLearner, train_dataloader: DataLoader,
          val_dataloader: Optional[DataLoader] = None,
          eval_every: int = 50, total_iterations: int = 1000,
          disc_only: bool = True):

    cur_iter = 0
    cur_loss_G = 0.0
    cur_loss_D = 0.0
    cur_samples = 0
    train_loss_G = []
    train_loss_D = []
    val_loss_G = []
    val_loss_D = []
    pbar = None

    while True:
        for batch in train_dataloader:
            if pbar is None:
                pbar = tqdm(total=eval_every)
            bs = len(batch[0])
            lossD, _ = learner.train_iter(batch, True)
            cur_loss_D += lossD * bs
            # if not only_disc:
            #     cur_loss_G += model.loss_G.item() * len(L)
            cur_samples += bs
            pbar.update(1)
            cur_iter += 1
            # actions for current 'eval every'
            if cur_iter % eval_every == 0:
                pbar.close()
                pbar = None
                train_loss_D.append(cur_loss_D / cur_samples)
                print(f'Discriminator train loss: {train_loss_D[-1]:.2e}')
                cur_loss_D = 0.0
                if not disc_only:
                    train_loss_G.append(cur_loss_G / cur_samples)
                    print(f'Generator train loss: {train_loss_G[-1]:.2e}')
                    cur_loss_G = 0.0
                cur_samples = 0

            if cur_iter >= total_iterations:
                if pbar is not None:
                    pbar.close()
                return (np.array(train_loss_D), np.array(val_loss_D),
                        np.array(train_loss_G), np.array(val_loss_G))


if __name__ == '__main__':
    main()
