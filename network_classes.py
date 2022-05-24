import torch
import torch.nn as nn


class EnergyNet(torch.nn.Module):
    def __init__(self):
        super(EnergyNet, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(1000, 750),
                                     nn.ReLU(),
                                     nn.Linear(750, 500),
                                     nn.BatchNorm1d(500),
                                     nn.LeakyReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(500, 250),
                                     nn.ReLU(),
                                     nn.Linear(250,125),
                                     nn.LeakyReLU(),
                                     nn.Linear(125, 50)
                                     )
        self.layer1 = nn.Sequential(nn.Linear(50, 400),
                                    nn.LeakyReLU(),
                                    nn.Linear(400, 750),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(750, 400),
                                    nn.BatchNorm1d(400),
                                    nn.LeakyReLU(),
                                    nn.Linear(400, 200),
                                    nn.LeakyReLU()
                                    )
        self.fc = nn.Sequential(
            nn.Linear(200, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.layer1(x)
        x = self.fc(x)

        return x


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(1000, 500),
                                     nn.Dropout(p=0.5),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(500),
                                     nn.Linear(500, 100),
                                     nn.LeakyReLU(),
                                     )
        self.fc = nn.Sequential(
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)

        return x
