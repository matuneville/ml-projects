from torch import nn
import torch

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, kernel_size=3):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.middle = self.middle_block(128, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):
        """Forward pass"""

        # Downsampling
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Middle processing
        middle = self.middle(conv3)

        # Upsampling, adding skip connections
        upconv3 = self.upconv3(middle)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], dim=1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], dim=1))

        return upconv1.squeeze()

    def contract_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.1)
        )
        return contract

    def middle_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def expand_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Dropout(0.1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        return expand