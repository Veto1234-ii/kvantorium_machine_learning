import torch 
import torch.nn as nn

""" Transposed convolution = upsampling + convolution """

img = torch.Tensor([2, 5, 13, 20])

img = img.view(1, 1, 2, 2)

convTransposed = nn.ConvTranspose2d(in_channels  = 1,
                                    out_channels = 2,
                                    kernel_size  = 3,
                                    stride       = 2,
                                    padding      = 1)

out = convTransposed(img)

print(out)
 
"""Задание № 1
Создать тензор размера 4 x 4 к нему применить Transposed convolution
"""




