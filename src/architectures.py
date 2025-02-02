import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.block1 = self.conv_block(in_channels, 16)
        self.block2 = self.conv_block(16, 32)
        self.block3 = self.conv_block(32, 64)
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, inputs):
        x = self.pool(self.block1(inputs))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.block4(x)
        return F.interpolate(x, size=inputs.shape[2:], mode='bilinear', align_corners=False)

class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.bottleneck = self.conv_block(64, 128)
        self.dec3 = self.conv_block(128 + 64, 64)
        self.dec2 = self.conv_block(64 + 32, 32)
        self.dec1 = self.conv_block(32 + 16, 16)
        self.final = nn.Conv2d(16, 2, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        enc1, enc2, enc3 = self.enc1(x), self.enc2(self.pool(self.enc1(x))), self.enc3(self.pool(self.enc2(x)))
        x = self.bottleneck(self.pool(enc3))
        x = F.interpolate(x, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        x = F.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc1], dim=1)
        return self.final(x)
