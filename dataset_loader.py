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
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,)), ])
        self.img_inputs=[]
        self.img_target=[]
        print(" Loading Files .... ")
        for file in self.files:
            if file.find('input') != -1:
                path = os.path.join(self.path_Dataset, file)
                pil_image = Image.open(path)
                self.img_inputs.append(np.array(pil_image))



            elif file.find('target') != -1:
                path = os.path.join(self.path_Dataset, file)
                pil_image = Image.open(path)
                self.img_target.append(np.array(pil_image))


            else:
                print("the file name is incorrect, it need to include input or traget! ")
                quit()

        print(" Done loading Files!")

    def __getitem__(self, index):


        targets = self.transform(self.img_target[index])
        inputs = self.transform(self.img_inputs[index])

        return inputs, targets

    def __len__(self):
        return len(self.files)//2


if __name__ == "__main__":
    dataset = CustomDataset("D:/Uni_Ulm/oulu/VD_dataset")
    loader = DataLoader(dataset, batch_size=5)
    print(f"Dataset length: {len(dataset)}")
    x, y = next(iter(loader))
    for i,(x, y) in enumerate(loader):
        print(x.shape)
        print(y.shape)
        # plt.imshow(x[0].permute(1, 2, 0))
        # plt.show()
        plt.imshow(y[0].permute(1, 2, 0))
        plt.show()
        print(i)
