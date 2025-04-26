import torch.nn as nn

class ToneCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()

        # conv layers with frequency-only pooling
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((3,1), (3,1)),  # pool freq only

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2,1), (2,1)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2,1), (2,1)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2,1), (2,1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU()
        )

        # global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, n_classes) # 4 possible tones
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
