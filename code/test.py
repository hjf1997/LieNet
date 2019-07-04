import torch
import numpy as np
from pymanopt.manifolds import Rotations
from utils import egrad2rgrad, retr
import scipy.io as scio

W1 = scio.loadmat('W1.mat')
W1 = W1['W1_i']

W1_grad = scio.loadmat('W1grad.mat')
W1_grad = W1_grad['W1grad_i']

ro = Rotations(3, 1)
#print(ro.egrad2rgrad(W1_grad.T, W1.T,))
rgrad = egrad2rgrad(torch.from_numpy(W1.T).view(3,3,1), torch.from_numpy(W1_grad.T).view(3,3,1))
print(rgrad)
w = retr(torch.from_numpy(W1.T).view(3,3,1), rgrad.permute(1,0,2), 1)
print(w)
w = w.view(3, 3)
print(w.mm(w.t()))
