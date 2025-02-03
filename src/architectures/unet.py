import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, init_filters=16, num_classes=2):
        """
        U-Net for medical image segmentation
        
        Args:
            in_channels (int): Input channels (1 for grayscale, 3 for RGB)
            init_filters (int): Number of filters in first conv layer
            num_classes (int): Output classes (2 for binary segmentation)
        """
        super().__init__()
        
        # Encoder Path
        self.enc1 = self._conv_block(in_channels, init_filters)
        self.enc2 = self._conv_block(init_filters, init_filters * 2)
        self.enc3 = self._conv_block(init_filters * 2, init_filters * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(init_filters*4, init_filters*8)
        
        # Decoder Path
        self.dec3 = self._conv_block(init_filters*12, init_filters*4)  # 8*2 + 4 = 12?
        self.dec2 = self._conv_block(init_filters*6, init_filters*2)
        self.dec1 = self._conv_block(init_filters*3, init_filters)
        
        # Final Output
        self.final = nn.Conv2d(init_filters, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_ch, out_ch):
        """Helper for convolutional blocks"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        bn = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = F.interpolate(bn, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)