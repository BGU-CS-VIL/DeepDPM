import torch.nn as nn


class ResNetTrunk(nn.Module):
    """
    Adapted from https://github.com/xu-ji/IIC/blob/master/
    """
    def __init__(self):
        super(ResNetTrunk, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,
                               track_running_stats=self.batchnorm_track),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            track_running_stats=self.batchnorm_track))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, track_running_stats=self.batchnorm_track))

        return nn.Sequential(*layers)