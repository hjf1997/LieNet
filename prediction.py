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
from utils import egrad2rgrad, retr

def train(datat, iter, train):

    if datat=='g3d':
        dataset = G3dDataset(train=train)
        trainloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    elif datat=='CSL':
        pass
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if train:
        net = LieNet(device)
        net.to(device)

        for epoch in range(iter):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):

                inputs = data['fea'].float().to(device)
                labels = data['label'].to(device)
                outputs = net(inputs)
                loss_ = loss.softmax_loss_LieNet(outputs, labels)
                running_loss += loss_
                net.zero_grad()
                loss_.backward()
                # print(net.rot1.w[:, :,1])
                # for i in range(1):
                #     param = eval('net.rot'+str(i+1)+'.w')
                #     grad = egrad2rgrad(param ,param.grad)
                #     net.rot1.w = retr(param, -0.001*grad, param.shape[-1])
                # print(net.rot1.w[:,:,1])
                _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                count = 0
                for param in net.parameters():
                    #if count == 3:
                        #print(param.data[1,:])
                    if count < 3:
                        count += 1
                        param.grad.data = egrad2rgrad(param.grad.data, param.data)
                        data = retr(param.data, -0.1*param.grad.data, param.data.shape[-1])
                        param.data = data
                    else:
                        count += 1
                        param.data.sub_(0.01 * param.grad.data)
            print(running_loss/(i+1))


train('g3d', 4000, True)




