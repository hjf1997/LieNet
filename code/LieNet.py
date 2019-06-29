import torch
import numpy as np
import torch.nn as nn
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LieNet(nn.Module):

    def __init__(self):
        super(LieNet, self).__init__()
        self.rot1 = RotMap(3,  342)
        self.pool1 = Pooling('space')
        self.rot2 = RotMap(3, 171)
        self.pool2 = Pooling('time')
        self.rot3 = RotMap(3, 171)
        self.pool3 = Pooling('time')
        self.log = LogMap()
        self.relu = Relu()
        self.fc1 = nn.Linear(4788, 20)

    def forward(self, x):
        x = self.rot1(x)
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)
        x = self.rot2(x)
        print(x.shape)
        x = self.pool2(x)
        print(x.shape)
        x = self.rot3(x)
        print(x.shape)
        x = self.pool3(x)
        print(x.shape)
        x = self.log(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x = self.fc1(x.transpose(1, 0))
        return x


class Pooling(torch.nn.Module):

    def __init__(self, pool):

        super(Pooling, self).__init__()
        self.pool = pool

    def cal_roc_angel(self, r):
        epsilon = 1e-12;
        mtrc = torch.trace(r)
        if torch.abs(mtrc - 3) <= epsilon:
            a = 0
        elif torch.abs(mtrc + 1) <= epsilon:
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
            assert x.shape[3] % 2 == 0
            Y = torch.zeros((x.shape[0], x.shape[1], x.shape[2], int(x.shape[3]/2), x.shape[4]))

            for i4 in range(x.shape[4]):
                for i0 in range(x.shape[0]):
                    for i3 in range(0,  x.shape[3], 2):
                        r_tt = x[i0, :, :, i3:i3+2, i4]
                        I = self.max_rot_angel(r_tt)
                        Y[i0, :, :, int(i3/2), i4] = r_tt[:, :, I]
        else:
            Y = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], math.ceil(x.shape[4] /4)))

            for i0 in range(x.shape[0]):
                for i3 in range(x.shape[3]):
                    for i4 in range(0, x.shape[4], 4):
                        if i4 + 3 < x.shape[4]:
                            r_tt = x[i0, :, :, i3, i4:i4+4]
                            I = self.max_rot_angel(r_tt)
                            Y[i0, :, :, i3, int(i4/4)] = r_tt[:, :, I]
                        else:
                            r_tt = x[i0, :, :, i3, i4:]
                            I = self.max_rot_angel(r_tt)
                            Y[i0, :, :, i3, -1] = r_tt[:, :, I]
        return Y


class LogMap(torch.nn.Module):

    def forward(self, x):
        """
        :param X: train or test  data with the dimension of [N ,D_in, D_in, num, frame]
        :return:
        """
        Y = torch.zeros((4 * x.shape[3] * x.shape[4], x.shape[0]))
        for i0 in range(x.shape[0]):
            for i3 in range(x.shape[3]):
                for i4 in range(x.shape[4]):
                    r_t = x[i0, :, :, i3, i4]
                    axis = torch.zeros(4)
                    axis[:3] = self.cal_roc_axis(r_t)
                    axis[3] = self.cal_roc_angel(r_t)
                    Y[i3 * 4 * x.shape[4] + i4 * 4:i3 * 4 * x.shape[4] + i4 * 4 + 4, i0] = axis.view(-1)
        return Y

    def cal_roc_angel(self, r):
        epsilon = 1e-12;
        mtrc = torch.trace(r)
        if torch.abs(mtrc - 3) <= epsilon:
            a = 0
        elif torch.abs(mtrc + 1) <= epsilon:
            a = np.pi
        else:
            a = torch.acos((mtrc - 1)/2)
        return a

    def cal_roc_axis(self, r):
        angel = self.cal_roc_angel(r)
        sin = torch.sin(angel)
        log = angel / (2 * sin) * (r - torch.t(r))
        # 下一行有点问题，与视觉slam， 对于反对称矩阵的fi1 2 3位置定义不一样
        fi = torch.tensor([log[2, 1], log[2, 0], log[1, 0]])
        fi = fi / torch.norm(fi)
        return fi


class Relu(torch.nn.Module):

    def forward(self, x):
        epslon = 0.3
        Y = torch.zeros_like(x)
        for j in range(x.shape[1]):
            for i in range(int(x.shape[0] / 4)):
                r_t = x[i * 4: i * 4 + 4, j]
                ir_t1 = torch.abs(r_t) < epslon
                ir_t2 = r_t < 0
                for k in range(r_t.shape[0]):
                    if ir_t1[k].item() == 1 and ir_t2[k].item() == 1:
                        r_t[k] = -epslon
                    elif ir_t1[k].item() == 1 and ir_t2[k].item() == 0:
                        r_t[k] = epslon
                Y[i * 4: i * 4 + 4, j] = r_t
        return Y


class RotMap(torch.nn.Module):

    def __init__(self, D_in, num):

        super(RotMap, self).__init__()
        np.random.seed(1234)
        w = torch.nn.Parameter(torch.zeros(D_in, D_in, num))
        for i in range(num):
            a = torch.randn(D_in, D_in)
            u, s, v = torch.svd(a)
            w[:, :, i] = u
        self.w = torch.nn.Parameter(w)

    def forward(self, x):
        """
        :param x: train or test with the dimension of [N ,D_in, D_in, num, frame]
        :return:
        """
        for i in range(x.shape[0]):
            for j in range(x.shape[4]):
                for z in range(x.shape[3]):
                    x[i, :, :, z, j] = self.w[:, :, z].mm(x[i, :, :, z, j])
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = LieNet()
train = torch.randn(30,3,3,342,100).to(device)
print(net.parameters())
net.to(device)
y = net(train)


