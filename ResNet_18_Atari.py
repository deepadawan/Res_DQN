
from torchvision.models.resnet import resnet18
import torch.nn as nn

def resnet(num_classes=4):
    """Adaptation of Resnet18 for ImageNet classification to Atari images"""
    test = resnet18(num_classes = num_classes)
    test.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=3,
                               bias=False)
    test.avgpool = nn.AvgPool2d(6)
    test.name = "resnet"
    return test
