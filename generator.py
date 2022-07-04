import torch
import torch.nn as nn


class GenBlock(nn.Module):
    def __init__(self, input_channels, output_channels, down=True, activation="relu", use_dropout=False):
        super(GenBlock, self).__init__()

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

        if down:
            if activation == "relu":
                self.conv = nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
                    nn.ReLU()
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
                    nn.LeakyReLU()
                )
        else:
            if activation == "relu":
                self.conv = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1, bias=False),
                                          nn.BatchNorm2d(output_channels), nn.ReLU()
                                          )
            else:
                self.conv = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1, bias=False),
                                          nn.BatchNorm2d(output_channels), nn.LeakyReLU()
                                          )

    def forward(self, x):
        x = self.conv(x)

        if self.use_dropout:
            return self.dropout(x)
        else:
            return x


class Gen(nn.Module):
    def __init__(self, input_channels=3, features=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        ##### Down #####

        self.n1 = GenBlock(features, features * 2, down=True, activation="leaky", use_dropout=False)

        self.n2 = GenBlock(
            features * 2, features * 4, down=True, activation="leaky", use_dropout=False
        )
        self.n3 = GenBlock(features * 4, features * 8, down=True, activation="leaky", use_dropout=False
                           )
        self.n4 = GenBlock(features * 8, features * 8, down=True, activation="leaky", use_dropout=False
                           )
        self.n5 = GenBlock(features * 8, features * 8, down=True, activation="leaky", use_dropout=False
                           )
        self.n6 = GenBlock(features * 8, features * 8, down=True, activation="leaky", use_dropout=False
                           )
        self.final_n = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
                                     )

        ####### Up #######

        self.m1 = GenBlock(features * 8, features * 8, down=False, activation="relu", use_dropout=True)

        self.m2 = GenBlock(features * 8 * 2, features * 8, down=False, activation="relu", use_dropout=True
                           )
        self.m3 = GenBlock(features * 8 * 2, features * 8, down=False, activation="relu", use_dropout=True
                           )
        self.m4 = GenBlock(features * 8 * 2, features * 8, down=False, activation="relu", use_dropout=False
                           )
        self.m5 = GenBlock(features * 8 * 2, features * 4, down=False, activation="relu", use_dropout=False
                           )
        self.m6 = GenBlock(features * 4 * 2, features * 2, down=False, activation="relu", use_dropout=False
                           )
        self.m7 = GenBlock(features * 2 * 2, features, down=False, activation="relu", use_dropout=False)

        self.final_m = nn.Sequential(
            nn.ConvTranspose2d(features * 2, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        n1 = self.initial(x)
        n2 = self.n1(n1)
        n3 = self.n2(n2)
        n4 = self.n3(n3)
        n5 = self.n4(n4)
        n6 = self.n5(n5)
        n7 = self.n6(n6)
        final_n = self.final_n(n7)
        m1 = self.m1(final_n)
        m2 = self.m2(torch.cat([m1, n7], 1))
        m3 = self.m3(torch.cat([m2, n6], 1))
        m4 = self.m4(torch.cat([m3, n5], 1))
        m5 = self.m5(torch.cat([m4, n4], 1))
        m6 = self.m6(torch.cat([m5, n3], 1))
        m7 = self.m7(torch.cat([m6, n2], 1))
        final_m = self.final_m(torch.cat([m7, n1], 1))
        return final_m


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Gen(input_channels=3, features=64)
    pred = model(x)
    print(pred.shape)
