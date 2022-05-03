"""Original U Net.

code modified from:
    https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py

- Author: NamSahng
- Email: namskgyreen@naver.com
"""

import torch
from torch import nn


def crop_concat(feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
    """Crop feat1 tensor and concat feat2 tensor."""
    feat1_shape = feat1.shape[2:]
    feat2_shape = feat2.shape[2:]
    start_h = int((feat1_shape[0] - feat2_shape[0]) / 2)
    start_w = int((feat1_shape[1] - feat2_shape[1]) / 2)
    end_h = start_h + feat2_shape[0]
    end_w = start_w + feat2_shape[1]
    feat1 = feat1[:, :, start_h:end_h, start_w:end_w]
    return torch.cat([feat1, feat2], 1)


class ConvBlock(nn.Module):
    """Convolution Block."""

    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        """Initialize."""
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, middle_channels, kernel_size=3, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            middle_channels, out_channels, kernel_size=3, stride=1, padding=0
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv1(feat)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class Unet(nn.Module):
    """Original Unet."""

    def __init__(self, num_classes: int, input_channels: int = 1) -> None:
        """Initialize."""
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up4 = nn.ConvTranspose2d(
            nb_filter[4], nb_filter[3], kernel_size=2, stride=2
        )
        self.up3 = nn.ConvTranspose2d(
            nb_filter[3], nb_filter[2], kernel_size=2, stride=2
        )
        self.up2 = nn.ConvTranspose2d(
            nb_filter[2], nb_filter[1], kernel_size=2, stride=2
        )
        self.up1 = nn.ConvTranspose2d(
            nb_filter[1], nb_filter[0], kernel_size=2, stride=2
        )

        self.conv0_0 = ConvBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = ConvBlock(nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = ConvBlock(nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = ConvBlock(nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = ConvBlock(nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(crop_concat(x3_0, self.up4(x4_0)))
        x2_2 = self.conv2_2(crop_concat(x2_0, self.up3(x3_1)))
        x1_3 = self.conv1_3(crop_concat(x1_0, self.up2(x2_2)))
        x0_4 = self.conv0_4(crop_concat(x0_0, self.up1(x1_3)))

        output = self.final(x0_4)
        return output
