# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch.nn.functional as F
import norms
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


def get_norm(n_filters, norm):
    if norm is None:
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)
    elif norm == "act":
        return norms.ActNorm(n_filters, False)


class ConvNet(nn.Module):
    def __init__(self, input_channels=3, base_channels=16, depth=3, widen_factor=2, norm=None, leak=.2):
        super(ConvNet, self).__init__()
        k = widen_factor
        print('| ConvNet %dx%d' %(depth, k))

        self.layers = []

        self.layers.append(nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False))
        self.layers.append(nn.LeakyReLU(leak))

        for i in range(depth):
            self.layers.append(nn.Conv2d(base_channels * (k**i), base_channels * (k**(i+1)), 4, 2, 1, bias=False))
            self.layers.append(get_norm(base_channels * (k**(i+1)), norm))
            self.layers.append(nn.LeakyReLU(leak))

        self.net = nn.Sequential(*self.layers)
        self.last_dim = base_channels * (k**(i+1))
        #self.layers.append(nn.Conv2d(self.last_dim, 1, 4, 1, 0, bias=False))


    def forward(self, x, vx=None):
        out = self.net(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
