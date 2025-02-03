import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_channels=3, init_filters=16, num_classes=2):
        """
        Fully Convolutional Network (FCN) for semantic segmentation.
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            init_filters (int): Number of filters in the first convolutional layer.
            num_classes (int): Number of output classes (2 for binary segmentation).
        """
        super().__init__()
        
        # Blocks 1-3
        self.block1 = self._conv_block(in_channels, init_filters)
        self.block2 = self._conv_block(init_filters, init_filters * 2)
        self.block3 = self._conv_block(init_filters * 2, init_filters * 4)
        
        # Final block
        self.block4 = nn.Sequential(
            nn.Conv2d(init_filters * 4, init_filters * 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_filters * 4, num_classes, kernel_size=1)
        )
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_ch, out_ch):
        """
        Helper function to create a convolutional block.
        
        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        
        Returns:
            nn.Sequential: A convolutional block with Conv2d, BatchNorm, and ReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the FCN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, C, H, W].
        
        Returns:
            torch.Tensor: Output tensor of shape [N, num_classes, H, W].
        """
        # Encoder path
        x = self.block1(x)
        x = self.pool(x)
        
        x = self.block2(x)
        x = self.pool(x)
        
        x = self.block3(x)
        x = self.pool(x)
        
        # Final block
        x = self.block4(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=(x.shape[2]*8, x.shape[3]*8), mode='bilinear', align_corners=False)
        
        return x