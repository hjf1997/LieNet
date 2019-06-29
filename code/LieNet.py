import torch
import numpy as np
import torch.nn as nn


class LieNet(nn.Module):

    def __init__(self):
        super(LieNet, self).__init__()

    def forward(self, x):
        pass

    def relu(self, x):
        pass

    def pooling(self, x):
        pass

    def logmap(self, x):
        pass


class Pooling(torch.nn.Module):

    def __init__(self, pool):

        super(Pooling, self).__init__()
        self.pool = pool

    def cal_roc_angel(self, r):
        epsilon = 1e-12;
        mtrc = torch.trace(r)
        if torch.abs(mtrc - 3).numpy() <= epsilon:
            a = 0
        elif torch.abs(mtrc + 1).numpy() <= epsilon:
            a = np.pi
        else:
            a = torch.acos((mtrc - 1)/2)
        return a

    def max_rot_angel(self, r):
        m_r = 0
        i_r = 0
        for i in range(r.shape[2]):
            r_a = self.cal_roc_angel(r[:, :, i])
            if r_a > m_r:
                m_r = r_a
                i_r = i
        return i_r

    def forward(self, x):
        """

        :param x: train or test  data with the dimension of [N ,D_in, D_in, num, frame]
        :return:
        """

        if self.pool == 'space':
            Y = torch.zeros((x.shape[0], x.shape[1], x.shape[2], int(x.shape[3]/2), x.shape[4]))

            assert x.shape[3] / 2 == 0

            for i4 in range(x.shape[4]):
                for i0 in range(x.shape[0]):
                    for i3 in range(0, 2, x.shape[3]):
                        r_tt = x[i0, :, :, i3:i3+2, i4]
                        I = self.max_rot_angel(r_tt)
                        Y[i0, :, :, int(i3/2), i4] = r_tt[:, :, I]
        else:
            Y = torch.zeros((x.shape[0], x.shape[1], x.shape[2], int(x.shape[3]/4), x.shape[4]))





class Relu(torch.nn.Module):

    def forward(self, *input):
        pass


class RotMap(torch.nn.Module):

    def __init__(self, D_in, num):

        super(RotMap, self).__init__()
        np.random.seed(1234)
        w = np.zeros((D_in, D_in, num))
        for i in range(num):
            a = np.random.rand(D_in, D_in)
            u, s, v = np.linalg.svd(a)
            w[:, :, i] = u

        self.w = torch.from_numpy(w)

    def forward(self, x):
        """

        :param x: train or test with the dimension of [N ,D_in, D_in, num, frame]
        :return:
        """
        x_per = x.permute(1, 2, 3, 4, 0)  # [D_in, D_in, num, frame, N]
        Y = torch.zeros_like(x_per)
        for i in range(x_per.shape[2]):
            for j in range(x_per.shape[3]):
                for z in range(x_per.shape[4]):
                    Y[:, :, i, j, z] = self.w[:, :, i].mm(x_per[:, :, i, j, z])

        return Y.permute(4, 0, 1, 2, 3)






