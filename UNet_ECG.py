import torch
import torch.nn as nn
import torch.nn.functional as F

from UNet_ECG_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        super(UNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Down path
        self.inc = DoubleConv(n_channels, 4)
        self.down1 = Down(4, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.down4 = Down(32, 64)
        
        
        # Up path
        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.up3 = Up(16, 8)
        self.up4 = Up(8, 4)
        
        # Output
        self.outc = OutConv(4, n_classes)

    def forward(self, x):
        # Down path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
                
        # Up path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits

# Example usage
if __name__ == "__main__":
    # Create model
    model = UNet(n_channels=1, n_classes=4)
    
    # Test with random input
    input_tensor = torch.randn(1, 1, 5000)  # (batch_size, channels, signal_length)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Should be (1, 4, 1000)
    
    label = torch.argmax(output[0], dim=0)
    
    print(label.size())
    print(label)
    
    print(output[0][0])
    print(output[0][1])
    print(output[0][2])
    print(output[0][3])


