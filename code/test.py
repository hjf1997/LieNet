import torch
import numpy as np
from utils import egrad2rgrad, retr, multiprod
import scipy.io as scio
from pymanopt.manifolds import Rotations
import torch.nn.functional as F
import torch.nn as nn
W1 = scio.loadmat('W1.mat')
W1 = W1['W1_i']

W1_grad = scio.loadmat('W1grad.mat')
W1_grad = W1_grad['W1grad_i']

ro = Rotations(3, 1)
ro_grad = ro.egrad2rgrad(W1.T, W1_grad.T).T
print(ro_grad)
ro_retr = ro.retr(W1.T, ro_grad.T * -1).T
print(ro_retr)

rgrad = egrad2rgrad(torch.from_numpy(W1.T).view(3,3,1), torch.from_numpy(W1_grad.T).view(3,3,1)).permute(1,0,2)
print(rgrad.view(3,3))
w = retr(torch.from_numpy(W1.T).view(3,3,1), -1 * rgrad.permute(1,0,2), 1).permute(1,0,2)
print(w.view(3,3))
#print(w)
#w = w.view(3, 3)
#print(w.mm(w.t()))

# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         self.fc1 = nn.Linear(16*6*6,120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 20)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def num_flat_features(self,x):
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
#
#
# net = Net()
# input = torch.randn(30 , 1, 32, 32)
# out = net(input)
#
# target = (torch.ones(30) * 19).long()
# #target = target.view(1, -1)
# critetion = nn.CrossEntropyLoss()
#
# loss = critetion(out, target)
# print(loss)
# net.zero_grad()
# loss.backward()
# print('1')
