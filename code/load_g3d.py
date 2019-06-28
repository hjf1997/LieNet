# implemented by junfeng Hu
# 28/6/2019

import os
import scipy.io as scio
import torch
import numpy as np
from torch.utils.data import Dataset


def dataExtract(lie_data_raw):
    """
    It isn't convenient to extract .mat by python, you could use Matlab to look the struct of data
    :param lie_data_raw: raw data
    :return: the extracted data
    """

    lie_data = {}
    lie_data['id'] = lie_data_raw[0][0][1][0][0][0][0]
    tem = lie_data_raw[0][0][1][0][0][1][0]
    lie_data['name'] = [tem[i][0] for i in range(tem.shape[0])]
    lie_data['set'] = lie_data_raw[0][0][1][0][0][2][0]
    lie_data['label'] = lie_data_raw[0][0][1][0][0][3][0]
    lie_data['pooling_index'] = lie_data_raw[0][0][1][0][0][4][0]

    return lie_data

def convert4d(fea):
    """
    :param fea: A sample of skeleton pose with the dimension of length * wide * frames *  number of SO(3)
                        For this dataset, the number of SO(3) equals to A(2)(19) = 342 given one pose has 20 joints -> 19 bones
    :return: A normal matrix with the dimension of length * wide *  number of SO(3) * frames
    """
    fea4D = np.zeros((fea['fea'][0][0].shape[0], fea['fea'][0][0].shape[1],
                      fea['fea'].shape[0], fea['fea'][0][0].shape[2]))

    for i in range(fea['fea'].shape[0]):
        fea4D[:, :, i, :] = fea['fea'][i][0]

    return fea4D



class G3dDataset(Dataset):
    """G3dDataset"""

    def __init__(self, lie_data, train_index, test_index, train):
        """
        :param lie_data:  contains the path of each sample, the label as well as train or test indicator
        :param train_index: the ids of  samples belonging to train set
        :param train: train or test
        """
        self.lie_data = lie_data
        self.train_index = train_index
        self.test_index = test_index
        self.train = train

        if train:
            index = train_index
        else:
            index = test_index

        lie_path = lie_data['dataDir'] + os.sep + lie_data['name'][1]
        fea = scio.loadmat(lie_path)
        fea4D = convert4d(fea)

        feas = np.zeros((index.shape[0], fea4D.shape[0], fea4D.shape[1], fea4D.shape[2],  fea4D.shape[3]))
        labels = np.zeros((index.shape[0]))

        for i in range(index.shape[0]):
            liePath = lie_data['dataDir'] + os.sep + lie_data['name'][index[i]]
            fea = scio.loadmat(liePath)
            fea4D = convert4d(fea)
            feas[i, :, :, :, :] = fea4D
            labels[i] = lie_data['label'][index[i]]

        self.feas = feas
        self.labels = labels

    def __len__(self):
        return self.feas.shape[0]

    def __getitem__(self, idx):
        sample = {'fea': self.feas[idx, :, :, :, :], 'label': self.labels[idx]}
        return sample


dataDir = '..' + os.sep + 'data' + os.sep + 'g3d'
train_config = r'liedb_g3d_lie20_half_inter.mat'
lie_train_config = scio.loadmat(os.path.join(dataDir, train_config))
lie_data_raw = lie_train_config['lie_train']
lie_data = dataExtract(lie_data_raw)
lie_data['dataDir'] = dataDir

train_index = np.where(lie_data['set'] == 1)[0]
test_index = np.where(lie_data['set'] == 2)[0]

# train_index = train_index[np.random.permutation(train_index.shape[0])]
# #shuffle # there is no need to do this with pytorch

# g = G3dDataset(lie_data, train_index, test_index, False)
# sample = g[1]


