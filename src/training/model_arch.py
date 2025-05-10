# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    """
    Custom CNN model for binary image classification.
    
    Args:
        num_classes (int): Number of output classes (default: 1 for binary classification)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
    """
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(CustomCNN, self).__init__()

        def _make_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.conv1 = _make_block(3, 64)
        self.conv2 = _make_block(64, 128)
        self.conv3 = _make_block(128, 256)
        self.conv4 = _make_block(256, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)