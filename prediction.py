import torch
import numpy as np
import os
import sys
sys.path.append('./code')

from LieNet import LieNet
from torch.utils.data import DataLoader
import torch.optim as optim
from load_g3d import G3dDataset

def train(datat, iter, train):

    if datat=='g3d':
        dataset = G3dDataset(train=train)
        trainloader = DataLoader(dataset, batch_size=30, shuffle=True, num_workers=4)
    elif datat=='CSL':
        pass

    if train:
        net = LieNet()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(iter):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):

                inputs = data['fea'].float()
                print(type(inputs))
                labels = data['label']
                outputs = net(inputs)
                print(outputs.shape)






