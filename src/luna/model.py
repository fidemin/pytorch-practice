import math

from torch import nn


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool = nn.MaxPool3d(2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.max_pool(y)
        return y


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(in_channels)

        self.conv1 = ConvolutionBlock(in_channels, conv_channels)
        self.conv2 = ConvolutionBlock(conv_channels, conv_channels * 2)
        self.conv3 = ConvolutionBlock(conv_channels * 2, conv_channels * 4)
        self.conv4 = ConvolutionBlock(conv_channels * 4, conv_channels * 8)

        self.fcl = nn.Linear(2 * 3 * 3 * 8 * conv_channels, 2)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x):
        y = self.batch_norm(x)

        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)

        # flatten
        y = y.view(y.size(0), -1)

        linear_output = self.fcl(y)
        softmax = self.softmax(linear_output)
        return linear_output, softmax

    def _init_weights(self):
        for module in self.modules():
            if type(module) in {nn.Linear, nn.Conv3d}:
                nn.init.kaiming_normal_(
                    module.weight.data,
                    a=0,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(
                        module.weight.data
                    )
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(module.bias, -bound, bound)
