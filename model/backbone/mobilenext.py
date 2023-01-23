import torch
import torch.nn as nn
from model.utils.utils import make_divisible
from model.utils.baseline import ConvBNReLU
import math

def sg_block_extra(in_chan, out_chan, stride, expand_ratio):
    hidden_dim = int(in_chan * expand_ratio)
    conv = nn.Sequential(
        # dw
        nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=stride, padding=1, groups=in_chan, bias=False),
        nn.BatchNorm2d(in_chan),
        nn.ReLU6(inplace=True),
        # pw
        nn.Conv2d(in_chan, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(hidden_dim),
        # pw
        nn.Conv2d(hidden_dim, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_chan),
        nn.ReLU6(inplace=True),
    )
    return conv

class SandglassBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride, expand_ratio, identity_tensor_multiplier=1.0, keep_3x3=False):
        super(SandglassBlock, self).__init__()
        self.stride = stride
        self.use_identity = False if identity_tensor_multiplier==1.0 else True
        self.identity_tensor_channels = int(round(in_chan * identity_tensor_multiplier))
        hidden_dim = in_chan // expand_ratio
        if hidden_dim < out_chan /6.:
            hidden_dim = math.ceil(out_chan / 6.)
            hidden_dim = make_divisible(hidden_dim, 16)

        self.use_res_connect = self.stride == 1 and in_chan == out_chan

        layers = []
        # dw
        if expand_ratio == 2 or in_chan==out_chan or keep_3x3:
            layers.append(ConvBNReLU(in_chan, in_chan, kernel_size=3, stride=1, groups=in_chan))
        if expand_ratio != 1:
            # pw-linear
            layers.extend([
                nn.Conv2d(in_chan, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            ])
        layers.extend([
            # pw
            ConvBNReLU(hidden_dim, out_chan, kernel_size=1, stride=1, groups=1),
        ])
        if expand_ratio == 2 or in_chan==out_chan or keep_3x3 or stride==2:
            layers.extend([
                # dw-linear
                nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=stride, groups=out_chan, padding=1, bias=False),
                nn.BatchNorm2d(out_chan),
            ])
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.use_identity:
                identity_tensor= x[:,:self.identity_tensor_channels,:,:] + out[:,:self.identity_tensor_channels,:,:]
                out = torch.cat([identity_tensor, out[:,self.identity_tensor_channels:,:,:]], dim=1)
            else:
                out = x + out
            return out
        else:
            return out

class MobileNeXt(nn.Module):
    def __init__(self, 
                    num_classes=1000,
                    width_mult=1.0,
                    identity_tensor_multiplier=1.0,
                    round_nearest=8
                ):
        super(MobileNeXt, self).__init__()
        input_channel = 32
        last_channel = 1280
        # building first layer
        input_channel = make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, kernel_size=3, stride=2)]
        last_layer = int(self.last_channel / width_mult)
        sand_glass_setting = [
            # t, c,  b, s
            [2, 96,  1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 1],
            [6, 576, 4, 2],
            [6, 960, 2, 1],
            [6, last_layer, 1, 1],
        ]
        
        # building sand glass blocks
        for t, c, b, s in sand_glass_setting:
            output_channel = make_divisible(c * width_mult, round_nearest)
            for i in range(b):
                stride = s if i == 0 else 1
                features.append(
                    SandglassBlock(
                        input_channel, 
                        output_channel, 
                        stride,
                        expand_ratio = t, 
                        identity_tensor_multiplier = identity_tensor_multiplier, 
                        keep_3x3 = (b == 1 and s == 1 and i == 0)
                    )
                )
                input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.extras = nn.ModuleList([
            sg_block_extra(last_layer, 512, 2, 0.2),
            sg_block_extra(512, 256, 2, 0.25),
            sg_block_extra(256, 256, 2, 0.5),
            sg_block_extra(256, 128, 2, 0.5)
        ])
        self._initialize_weights()
    
    def forward(self, x):
        features = []
        for i in range(13):
            x = self.features[i](x)
        N = len(self.features[13].conv)
        for i in range(N - 2):
            x = self.features[13].conv[i](x)
        features.append(x)
        for i in range(N - 2, N):
            x = self.features[13].conv[i](x)

        for i in range(14, len(self.features) - 1):
            x = self.features[i](x)

        last = len(self.features) - 1
        N = len(self.features[last].conv)
        for i in range(N - 2):
            x = self.features[last].conv[i](x)
        features.append(x)
        for i in range(N - 2, N):
            x = self.features[19].conv[i](x)

        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)

        return tuple(features)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)