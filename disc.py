import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Disc(nn.Module):
    def __init__(self, input_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        input_channels = features[0]
        for f in features[1:]:
            if f == features[-1]:
                stride = 1
            else:
                stride = 2
            layers.append(
                ConvBlock(input_channels, f, stride=stride),
            )
            input_channels = f

        layers.append(
            nn.Conv2d(
                input_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x






if __name__ == "__main__":

    x = torch.randn((1, 3, 286, 286))
    y = torch.randn((1, 3, 286, 286))
    model = Disc()
    pred = model(x, y)
    print(model)
    print(pred.shape)