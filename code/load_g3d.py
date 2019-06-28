#implemented by junfeng Hu
#28/6/2019

import os
import scipy.io as scio
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def dataExract(lie_data_raw):

    lie_data = {}
    lie_data['id'] = lie_data_raw[0][0][1][0][0][0][0]
    tem = lie_data_raw[0][0][1][0][0][1][0]
    lie_data['name'] = [tem[i][0] for i in range(tem.shape[0])]
    lie_data['set'] = lie_data_raw[0][0][1][0][0][2][0]
    lie_data['label'] = lie_data_raw[0][0][1][0][0][3][0]
    lie_data['pooling_index'] = lie_data_raw[0][0][1][0][0][4][0]

    return lie_data

dataDir = r'../data/g3d'
train_config = r'liedb_g3d_lie20_half_inter.mat'
lie_train_config = scio.loadmat(os.path.join(dataDir, train_config))
lie_data_raw = lie_train_config['lie_train']
lie_data = dataExract(lie_data_raw)

print(lie_data['name'])
#train_index = np.where()