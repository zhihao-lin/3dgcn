#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
    return feature


class DGCNN(nn.Module):
    def __init__(self, class_num, k= 20):
        super(DGCNN, self).__init__()
        self.k = k

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace= True),
                                   nn.Conv2d(64, 64, kernel_size=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace= True))

        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace= True),
                                   nn.Conv2d(64, 64, kernel_size=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace= True))

        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace= True))


        self.linear1 = nn.Sequential(
            nn.Conv1d(64 * 3, 1024, kernel_size= 1), 
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace= True)
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(1024 + 64 * 3 + 16, 256, kernel_size= 1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(inplace= True), 
            nn.Conv1d(256, 256, kernel_size= 1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(inplace= True),
            nn.Conv1d(256, 128, kernel_size= 1), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace= True),
            nn.Conv1d(128, class_num, kernel_size= 1) 
        )

    def forward(self, x, onehot):
        x = x.permute(0, 2, 1) #(bs, 3, point_num)
        point_num = x.size(2)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat([x1, x2, x3], dim= 1)
        x = self.linear1(x)
        x = x.max(-1)[0]
        x = x.unsqueeze(2).repeat(1, 1, point_num)
        onehot = onehot.unsqueeze(2).repeat(1, 1, point_num)

        x = torch.cat([x, x1, x2, x3, onehot], dim= 1)
        x = self.classifier(x)
        x = x.transpose(1, 2)
        return x

def test():
    from test import test_model
    model = DGCNN(50)
    dataset = "../../shapenetcore_partanno_segmentation_benchmark_v0"
    test_model(model, dataset, cuda= "0", bs= 4, point_num= 1024)

if __name__ == "__main__":
    test()