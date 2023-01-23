import torch.nn as nn
from model.utils.utils import make_divisible
from model.utils.baseline import Conv1x1BNReLU6, Conv3x3BNReLU6, ConvBNReLU
import math

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_dim, out_dim, stride, is_bias=False, onnx_compatible=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.identity = in_dim == out_dim and stride == 1
        self.batchNorm1 = nn.BatchNorm2d(in_dim)
        self.batchNorm2 = nn.BatchNorm2d(out_dim)
        self.depthwise = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=stride, groups=in_dim, bias=is_bias)
        self.pointwise = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=is_bias)
        self.relu = nn.ReLU() if onnx_compatible else nn.ReLU6(inplace=True)
    
    def forward(self, x):
        y = self.depthwise(x)
        y = self.batchNorm1(y)
        y = self.relu(y)
        y = self.pointwise(y)
        y = self.batchNorm2(y)
        return y if not self.identity else x + y

class InvertedResidual(nn.Module): # Identity
    def __init__(self, in_dim, out_dim, stride, expand_dim, is_bias=False, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        hidden_dim = round(in_dim * expand_dim)
        self.identity = in_dim == out_dim and stride == 1
        self.pointwise1 = nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=is_bias)
        self.batchNorm1 = nn.BatchNorm2d(hidden_dim)
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=is_bias)
        self.batchNorm2 = nn.BatchNorm2d(hidden_dim)
        self.pointwise2 = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=is_bias)
        self.batchNorm3 = nn.BatchNorm2d(out_dim)
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
        return y if not self.identity else x + y

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., onnx_compatible=False):
        super(MobileNetV2, self).__init__()
        self.cfgs = [
            # t(Expand Ratio), c(channel), n(Layer Number), s(Stride)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2], # Classifier Layer
            [6, 320, 1, 1], # Classifier Layer
        ]
        input_channel = make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [
            Conv3x3BNReLU6(3, input_channel, 2)
        ]
        for t, c, n, s in self.cfgs:
            output_channel = make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                stride = s if i == 0 else 1
                if t == 1:
                    layers.append(DepthwiseSeparableConv(input_channel, output_channel, stride))
                else:
                    layers.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        self.last_channel = int(1280 * max(1.0, width_mult))
        layers.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*layers)
        self.extras = nn.ModuleList([
            InvertedResidual(1280, 512, 2, 0.2),
            InvertedResidual(512, 256, 2, 0.5),
            InvertedResidual(256, 256, 2, 0.5),
            InvertedResidual(256, 64, 2, 0.5)
        ])
        self._initialize_weights()
        
    def forward(self, x):
        features = []
        for i in range(14):
            x = self.features[i](x)
        features.append(x)
        for i in range(14, len(self.features)):
            x = self.features[i](x)
        features.append(x)
        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)
        return tuple(features)

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