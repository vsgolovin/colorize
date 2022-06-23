import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from generators import UNet
from critics import SimpleCritic
from dataset_utils import ColorizationFolderDataset, tensor2image
from gan_learner import GANLearner


BATCH_SIZE = 24
GEN_WEIGHTS = 'checkpoints/resnet34.pth'
DISCR_WEIGHTS = 'checkpoints/discriminator.pth'
TRAIN_DATA = 'data/train'
VAL_DATA = 'data/val'
OUTPUT_DIR = 'output'
CIE_LAB = True  # RGB not yet supported
PIXEL_LOSS_WEIGHT = 100.
EXPORT_IMAGES = 64
SAVE_EVERY = 500
TOTAL_ITERATIONS = 5000


def main():
    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load pretrained generators
    net_G = UNet(
            resnet_layers=34,
            cie_lab=CIE_LAB
        ).to(device)
    net_G.load_state_dict(torch.load(GEN_WEIGHTS, map_location=device))
    net_G.freeze_encoder()

    # initialize discriminator
    net_D = SimpleCritic(nc_first=256).to(device)
    net_D.load_state_dict(torch.load(DISCR_WEIGHTS, map_location=device))

    # load dataset
    train_dataset = ColorizationFolderDataset(
        folder=TRAIN_DATA,
        transforms=T.Compose([
            T.RandomResizedCrop(224),
        ]),
        cie_lab=CIE_LAB
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = ColorizationFolderDataset(
        folder=VAL_DATA,
        cie_lab=CIE_LAB
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # train GAN
    learner = GANLearner(net_G, net_D, device=device, pixel_loss=F.l1_loss,
                         pixel_loss_weight=PIXEL_LOSS_WEIGHT,
                         gen_opt_params=dict(lr=1e-4, betas=(0.5, 0.99)),
                         discr_opt_params=dict(lr=2e-4, betas=(0.5, 0.99)))
    num_iter = 0
    while True:
        for batch in train_dataloader:
            lossD, lossG, _ = learner.train_iter(batch, False)
            print(f'\r{lossD}, {lossG}', end='')
            num_iter += 1

            # export images
            if num_iter % SAVE_EVERY == 0:
                torch.save(learner.net_G.state_dict(),
                           os.path.join(OUTPUT_DIR, f'gen_{num_iter}.pth'))
                exported = 0
                for batch in val_dataloader:
                    bs = len(batch[0])
                    learner.eval_iter(batch)
                    for i in range(min(EXPORT_IMAGES - exported, bs)):
                        fakes = learner.fake_images
                        img = tensor2image(fakes[i, 1:], L=fakes[i, 0:1],
                                           cie_lab=CIE_LAB)
                        fname = f'{exported}_{num_iter}.jpeg'
                        img.save(os.path.join(OUTPUT_DIR, fname))
                        exported += 1
                    if exported >= EXPORT_IMAGES:
                        break

            if num_iter == TOTAL_ITERATIONS:
                print()
                return


if __name__ == '__main__':
    main()
