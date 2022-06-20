from typing import Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataset_utils import LabColorizationDataset
from critic import net_D
from loss_functions import GANLoss
from tqdm import tqdm

# Folder with RGB original pictures
REAL_IMAGE_FOLDER = '../data/benchmarks/Original'
# Folder with RGB pictures generated by the latest version of the generator
FAKE_INAGE_FOLDER = '../data/benchmarks/ECCV16'
BATCH_SIZE = 32
EPOCHS = 10


class Discriminator(nn.Module):
    def __init__(self, net_D: nn.Module, lr: float = 2e-4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_D = net_D.to(self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr)

    def setup_input(self, data):
        self.real_L = data['real_L'].to(self.device)
        # self.fake_L = data['fake_L'].to(self.device) # Не нужно
        self.real_ab = data['real_ab'].to(self.device)
        self.fake_ab = data['fake_ab'].to(self.device)

    def backward_D(self):
        fake_image = torch.cat([self.real_L, self.fake_ab], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.real_L, self.real_ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def optimize(self):
        self.net_D.train()
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()


def train_discriminator(disc: nn.Module, train_dataloader: DataLoader,
                        val_dataloader: Optional[DataLoader] = None, epochs: int = 10):
    for e in range(epochs):
        cur_loss = 0.0
        tot_iter = 0
        for data in tqdm(train_dataloader):
            disc.setup_input(data)
            disc.optimize()
            cur_loss += disc.loss_D.item() * len(data['real_L'])
            tot_iter += len(data['real_L'])
        print(f'epoch: {e}')
        print(f'train_loss: {cur_loss / tot_iter}')


def main():
    net = net_D()
    disc = Discriminator(net_D=net)
    train_dataset = LabColorizationDataset(
            real_image_folder=REAL_IMAGE_FOLDER,
            fake_image_folder=FAKE_INAGE_FOLDER,
            transforms=T.Compose([T.Resize((224, 224))]),
        )  # No random transforms!!!
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_discriminator(disc, train_dataloader, epochs=EPOCHS)


if __name__ == '__main__':
    main()