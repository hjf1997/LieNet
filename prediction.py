import torch
import numpy as np
import os
import sys
sys.path.append('./code')

from LieNet import LieNet
from torch.utils.data import DataLoader
import torch.optim as optim
from load_g3d import G3dDataset
import loss

def train(datat, iter, train):

    if datat=='g3d':
        dataset = G3dDataset(train=train)
        trainloader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=0)
    elif datat=='CSL':
        pass

    if train:
        net = LieNet()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(iter):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):

                inputs = data['fea'].float()
                labels = data['label']
                outputs = net(inputs)
                loss_ = loss.softmax_loss_LieNet(outputs, labels)
                loss_.backward()
                for param in net.parameters():
                    print(param.grad)


train('g3d', 1, True)




