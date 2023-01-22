import torch.nn as nn
from ..utils.utils import make_divisible
from ..utils.baseline import Conv1x1BNReLU6, Conv3x3BNReLU6
import math

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_dim, out_dim, stride, onnx_compatible=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.batchNorm1 = nn.BatchNorm2d(in_dim)
        self.batchNorm2 = nn.BatchNorm2d(out_dim)
        self.depthwise = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=stride, groups=in_dim)
        self.pointwise = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU() if onnx_compatible else nn.ReLU6(inplace=True)
    
    def forward(self, x):
        y = self.depthwise(x)
        y = self.batchNorm1(y)
        y = self.relu(y)
        y = self.pointwise(y)
        y = self.batchNorm2(y)
        return y

class InvertedResidual(nn.Module): # Identity
    def __init__(self, dim, stride, expand_dim,onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(dim * expand_dim)
        self.pointwise1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(hidden_dim)
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim)
        self.batchNorm2 = nn.BatchNorm2d(hidden_dim)
        self.pointwise2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
        self.batchNorm3 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU() if onnx_compatible else nn.ReLU6(inplace=True)
        self.relu2 = nn.ReLU() if onnx_compatible else nn.ReLU6(inplace=True)
    
    def forward(self, x):
        y = self.pointwise1(x)
        y = self.batchNorm1(y)
        y = self.relu1(y)
        y = self.depthwise(y)
        y = self.batchNorm2(y)
        y = self.relu2(y)
        y = self.pointwise2(y)
        y = self.batchNorm3(y)
        return x + y

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., onnx_compatible=False):
        super(MobileNetV2, self).__init__()
        self.cfgs = [
            # t(Expand Ratio), c(channel), n(Layer Number), s(Stride)
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [
            Conv3x3BNReLU6(3, input_channel, 2)
        ]
        for t, c, n, s in self.cfgs:
            output_channel = make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                if t == 1:
                    layers.append(DepthwiseSeparableConv(input_channel, output_channel, s if i == 0 else 1))
                else :
                    assert input_channel == output_channel
                    layers.append(InvertedResidual(input_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.conv = Conv1x1BNReLU6(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()