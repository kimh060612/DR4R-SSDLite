import torch.nn as nn

class Conv3x3BNReLU6(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(Conv3x3BNReLU6, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchNorm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        return self.relu(x)
    
class Conv1x1BNReLU6(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv1x1BNReLU6, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchNorm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        return self.relu(x)

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding_sz = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding_sz, groups=groups, bias=True)
        self.bnlayer = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bnlayer(x)
        x = self.relu(x)
        return x