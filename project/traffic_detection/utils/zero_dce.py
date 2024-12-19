import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroDCE(nn.Module):
    def __init__(self):
        super(ZeroDCE, self).__init__()
        self.iterations = 4
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=32*2, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=32*2, out_channels=32, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(in_channels=32*2, out_channels=24, kernel_size=(3, 3),
                               stride=1, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)
  
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        x_r = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image_1, enhance_image
