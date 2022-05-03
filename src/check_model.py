"""Dummy test for U-net."""

import torch
from torch.nn import functional as F

from models import unet

if __name__ == "__main__":

    dummy = torch.ones((4, 1, 572, 572), dtype=torch.float)
    print(f"Input shape: {dummy.shape}")

    # Binary Segmentation
    model1 = unet.Unet(2)
    model1.eval()
    with torch.no_grad():
        dummy_output = model1.forward(dummy)
        dummy_output = F.softmax(dummy_output, dim=1)
        print("Binary Segmentation")
        print(f"Output shape: {dummy_output.shape}")
        print(f"output pixel sum: {dummy_output[0, :, 1, 1].numpy().sum()}")
        print()
        # log output and do NLL Loss for training

    # Multi-Class Segmentation
    model2 = unet.Unet(5)
    model2.eval()
    with torch.no_grad():
        dummy_output = model2.forward(dummy)
        dummy_output = F.softmax(dummy_output, dim=1)
        print("Multi-Class Segmentation")
        print(f"Output shape: {dummy_output.shape}")
        print(f"output pixel sum: {dummy_output[0, :, 1, 1].numpy().sum()}")
        print()
        # log output and do NLL Loss for training

    # Binary Segmentation
    model3 = unet.Unet(1)
    model3.eval()
    with torch.no_grad():
        dummy_output = model3.forward(dummy)
        dummy_output = torch.sigmoid(dummy_output).squeeze()
        print("Binary Segmentation")
        print(f"Output shape: {dummy_output.shape}")
        print()
        # do BCE Loss for training
