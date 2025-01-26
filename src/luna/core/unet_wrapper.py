import math

from torch import nn

from src.luna.core.unet import UNet


class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_layer = nn.BatchNorm2d(kwargs["in_channels"])
        self.unet = UNet(**kwargs)

        # result is between 0 and 1 (probability)
        self.final_layer = nn.Sigmoid()

        self._init_weights()

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

    def forward(self, x):
        x = self.input_layer(x)
        x = self.unet(x)
        y = self.final_layer(x)
        return y
