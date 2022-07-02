import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from generator import Gen
from disc import Disc
from dataset_loader import CustomDataset
import torch.utils.data as data


def train(output_path, train_dl, device, gen, disc, val_x, val_y, l1_lambda=100.0, learning_rate=2e-4, num_epochs=10):
    l1 = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()

    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999), )
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    Disc_losses = Gen_losses = Gen_GAN_losses = Gen_L1_losses = []

    for epoch in range(num_epochs):
        # train
        print(f"---- Epoch: {epoch + 1} ----")
        for batch, (inputs, tragets) in enumerate(train_dl):
            train_x, train_y = inputs.to(device), tragets.to(device)

            y_fake = gen(train_x)
            d_real = disc(train_x, train_y)
            d_real_loss = bce(d_real, torch.ones_like(d_real))
            d_fake = disc(train_x, y_fake.detach())
            d_fake_loss = bce(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            opt_disc.step()

            opt_disc.zero_grad()

            d_fake = disc(train_x, y_fake)
            G_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            L1_LOSS = l1(y_fake, train_y) * l1_lambda
            G_loss = G_fake_loss + L1_LOSS
            G_loss.backward()
            opt_gen.step()
            opt_gen.zero_grad()

            print(
                'Epoch [{}/{}], Step [{}/{}], disc_loss: {:.4f}, gen_loss: {:.4f},Disc(real): {:.2f}, Disc(fake):{:.2f}, gen_loss_gan:{:.4f}, gen_loss_L1:{:.4f}'.format(
                    epoch + 1, num_epochs, batch + 1, len(train_dl), d_loss.item(), G_loss.item(), y_fake.mean(),
                    d_fake.mean(), G_fake_loss.item(), L1_LOSS.item()))

            Gen_losses.append(G_loss.item())
            Disc_losses.append(d_loss.item())
            Gen_GAN_losses.append(G_fake_loss.item())
            Gen_L1_losses.append(L1_LOSS.item())

        with torch.no_grad():
            gen.eval()
            fk_batch = gen(val_x.to(device))

        save_prediction_results(output_path, "input images", "predicted images", "ground truth", epoch, val_x, fk_batch,
                                val_y)


def save_prediction_results(folder_path, input_title, predicted_title, ground_truth_title, epoch, batch1, batch2,
                            batch3=None):

    plt.figure(figsize=(40, 40))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title(input_title,fontsize=24)
    plt.imshow(np.transpose(vutils.make_grid(batch1, nrow=1, padding=5,
                                             normalize=True).cpu(), (1, 2, 0)))

    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title(predicted_title,fontsize=24)
    plt.imshow(np.transpose(vutils.make_grid(batch2, nrow=1, padding=5,
                                             normalize=True).cpu(), (1, 2, 0)))


    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title(ground_truth_title,fontsize=24)
    plt.imshow(np.transpose(vutils.make_grid(batch3, nrow=1, padding=5,
                                             normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(folder_path, "Pix2Pix-" + str(epoch) + ".png"))
    plt.close()


def main():
    num_epoch = 50
    num_workers = 0
    batch_size = 8
    l1_lambda = 100.0
    learning_rate = 2e-4
    dataset_dir = "D:/Uni_Ulm/oulu/VD_dataset"
    output_dir = "D:/Uni_Ulm/oulu/results"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device: {device}, batch: {batch_size}, epoch: {num_epoch}, number of worker : {num_workers}, "
          f"learning rate: {learning_rate}, l1 lambda: {l1_lambda}")

    gen = Gen(input_channels=3, features=64).to(device)
    disc = Disc(input_channels=3).to(device)

    train_dataset = CustomDataset(dataset_dir)


    # Random split
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size
    train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size])

    print(len(train_dataset))
    print(len(train_set))
    print(len(valid_set))

    train_dl = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_dl = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    val_x, val_y = next(iter(val_dl))
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    print(f"Train: {len(train_dl)}, val: {len(val_dl)}")
    train(output_dir, train_dl, device, gen, disc, val_x, val_y, l1_lambda=l1_lambda, learning_rate=learning_rate,
          num_epochs=num_epoch)


if __name__ == "__main__":
    main()
