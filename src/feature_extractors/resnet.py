"""
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
Was adapted from https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/
"""
import torch.nn as nn
import torchvision.models as models


def resnet50():
    backbone = models.__dict__['resnet50']()
    backbone.fc = nn.Identity()
    return {'backbone': backbone, 'dim': 2048}