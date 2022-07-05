import os

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib import pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, path_Dataset):
        self.path_Dataset = path_Dataset
        self.files = os.listdir(self.path_Dataset)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(512),
                                             transforms.ToTensor() ])

    def __getitem__(self, index):
        img_file = self.files[index]
        img_path = os.path.join(self.path_Dataset, img_file)
        image = np.array(Image.open(img_path))

        w = image.shape[1]

        inputs = self.transform(image[:, :w//2, :])
        targets = self.transform(image[:, w//2:, :])

        return inputs, targets

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    dataset = CustomDataset("D:/Uni_Ulm/oulu/maps/maps/train")
    loader = DataLoader(dataset, batch_size=5)
    print(f"Dataset length: {len(dataset)}")

    for i, (x, y) in enumerate(loader):
        print(x.shape)
        print(y.shape)
        plt.imshow(x[0].permute(1, 2, 0))
        plt.show()
        plt.imshow(y[0].permute(1, 2, 0))
        plt.show()
        print(i)
