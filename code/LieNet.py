# implemented by junfeng Hu
import torch
import numpy as np
import torch.nn as nn
import math


def so3Check(r): #  [N ,D_in, D_in, num, frame]
    """
    This function check whether tensor is on SO(3) to prevent errors. If any
    matrix of the tensor isn't a SO(3) a assert will throw.
    :param r:
    :return: None
    """
    for i0 in range(r.shape[0],10):
        for i3 in range(r.shape[3],100):
            for i4 in range(r.shape[4],30):
                k = r[i0,:,:, i3, i4].mm(r[i0,:,:, i3, i4].t())
                dig = k[0, 0] + k[1, 1] +k[2, 2]
                d = abs(dig.item() - 3.)
                if d > 0.01:
                    print("Sth goes wrong: i0={0} i3={1} i4={2} dig={3}".format(i0, i3, i4, dig))
                    assert d <= 0.01


class LieNet(nn.Module):

    def __init__(self, device):
        super(LieNet, self).__init__()
        self.rot1 = RotMap(3,  342)
        self.pool1 = Pooling('space')
        self.rot2 = RotMap(3, 171)
        self.pool2 = Pooling('time')
        self.rot3 = RotMap(3, 171)
        self.pool3 = Pooling('time')
        self.log = LogMap()
        self.relu = Relu(device)
        self.fc1 = nn.Linear(4104, 20)
        self.softmax = nn.Softmax(1)

    def forward(self, x):

        x = self.rot1(x)
        #so3Check(x)
        x = self.pool1(x)
        x = self.rot2(x)
        x = self.pool2(x)
        x = self.rot3(x)
        x = self.pool3(x)
        x = self.log(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.softmax(x)

        return x


class Pooling(torch.nn.Module):

    def __init__(self, pool):

        super(Pooling, self).__init__()
        self.pool = pool

    def cal_roc_angel(self, r):
        """
        Calculate euler angle of each matrix in tensor r. This is non-batch version
        of the function. Dismissed due to slow speed.
        :param r:
        :return: a
        """
        epsilon = 1e-12;
        mtrc = torch.trace(r)  # need about 9 seconds
        if torch.abs(mtrc - 3) <= epsilon:
            a = 0
        elif torch.abs(mtrc + 1) <= epsilon:
            a = np.pi
        else:
            a = torch.acos((mtrc - 1)/2)  # need little time
        return a

    def max_rot_angel(self, r):
        """
        Find max matrix of tensor r with maximum euler angle. This is non-batch version
        of the function. Dismissed due to slow speed.
        :param r:
        :return:
        """
        m_r = 0
        i_r = 0
        for i in range(r.shape[2]):
            r_a = self.cal_roc_angel(r[:, :, i])
            if r_a > m_r:
                m_r = r_a
                i_r = i
        return i_r

    def cal_roc_angel_batch(self, r):
        """
        Calculate euler angle of each matrix in tensor r.
        :param r: SO(3) tensor with the dimension of  [frame, N,  num, D_in, D_in]
        :return: euler angle with shape [ frame, N,  num]
        """
        epsilon = 1e-12;

        tr = BatchTrace()
        mtrc = tr(r)

        maskpi = (torch.abs(mtrc + 1) <= epsilon) * (torch.abs(mtrc - 3) > epsilon)
        mtrcpi = mtrc * maskpi.float() * np.pi

        maskacos = (torch.abs(mtrc + 1) > epsilon) * (torch.abs(mtrc - 3) > epsilon)
        mtrcacos = torch.acos((mtrc-1)/2) * maskacos.float()

        return mtrcpi + mtrcacos

    def max_rot_angel_batch(self, r, type): # frame, N,  num, D_in, D_in
        """
        Get matrix with maximum euler angle. Note that type means the kind of pooling
        2 means space and 4 means time.
        :param r: The tensor of dimension [frame, N,  num, D_in, D_in] or [num, N,  frame, D_in, D_in]
                        for space and time separate.
        :param type: 2 means space pooling and 4 means time pooling.
        :return: index with shape  [frame, N,  num/2] or [num, N, frame/4]
        """
        r_a = self.cal_roc_angel_batch(r)
        r_a = r_a.view(r_a.shape[0], r_a.shape[1], int(r_a.shape[2]/type), type)
        r_a, i = torch.max(r_a, 3)
        return i

    def forward(self, x):
        """
        :param x: train or test  data with the dimension of [N ,D_in, D_in, num, frame]
        :return:
        """
        if self.pool == 'space':
            assert x.shape[3] % 2 == 0

            x = x.permute(4, 0, 3, 1, 2)  # frame, N,  num, D_in, D_in
            shape = x.shape
            I = self.max_rot_angel_batch(x, 2)  # frame, N,  num/2
            I = I.view(-1)  # frame * N * num/2
            x = x.view(x.shape[0] * x.shape[1] * int(x.shape[2] / 2), 2, x.shape[3], x.shape[4])
            # frame * N * num/2, 2, D_in, D_in
            Y = x[[i for i in range(x.shape[0])], I, :, :]  # frame * N * num/2,  D_in, D_in
            Y = Y.view(shape[0], shape[1], int(shape[2]/2), shape[3], shape[4])  # frame , N , num/2,  D_in, D_in
            Y = Y.permute(1, 3, 4, 2, 0)  # N, D_in, D_in, num/2, frame

        else:
            Y = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], math.ceil(x.shape[4] /4)))
            # batch version
            x = x.permute(3, 0, 4, 1, 2)  # num, N,  frame, D_in, D_in
            x_ = x[:, :, :x.shape[2]-x.shape[2]%4, :, :]
            shape = x_.shape
            I = self.max_rot_angel_batch(x_, 4)
            I = I.view(-1)  # num, N, frame/4
            x_ = x_.contiguous().view(x_.shape[0] * x_.shape[1] * int(x_.shape[2] / 4), 4, x_.shape[3], x_.shape[4])
            Y = x_[[i for i in range(x_.shape[0])], I, :, :]
            Y = Y.view(shape[0], shape[1], int(shape[2]/4), shape[3], shape[4])  # num , N , frame/4,  D_in, D_in
            Y = Y.permute(1, 3, 4, 0, 2)

        return Y


class LogMap(torch.nn.Module):

    def forward(self, x):
        """
        :param X: train or test  data with the dimension of [N ,D_in, D_in, num, frame]
        :return:
        """
        # batch version
        x = x.permute(3, 4, 0, 1, 2)  # [ num, frame, N ,D_in, D_in]
        asix = self.cal_roc_axis_batch(x)  # [ num, frame, N, 3]
        angel = self.cal_roc_angel_batch(x).unsqueeze(3)  # num, frame, N,1
        Y = torch.cat([asix, angel], dim=3)
        Y = Y.permute(2, 0, 1, 3)  # N, num, frame, 4
        Y = Y.contiguous().view(Y.shape[0], -1)
        return Y

    def cal_roc_angel_batch(self, r): # frame, N,  num, D_in, D_in
        """
        Calculate euler angle of each matrix in tensor r.
        :param r: SO(3) tensor with the dimension of  [frame, N,  num, D_in, D_in]
        :return: euler angle with shape [ frame, N,  num]
        """

        epsilon = 1e-12;
        tr = BatchTrace()
        mtrc = tr(r)

        maskpi = (torch.abs(mtrc + 1) <= epsilon) * (torch.abs(mtrc - 3) > epsilon)
        mtrcpi = mtrc * maskpi.float() * np.pi

        maskacos = (torch.abs(mtrc + 1) > epsilon) * (torch.abs(mtrc - 3) > epsilon)
        mtrcacos = torch.acos((mtrc-1)/2) * maskacos.float()
        j = torch.isnan(mtrcacos).sum().item()

        # check nan in matrix
        assert j != 0

        return mtrcpi + mtrcacos

    def cal_roc_axis_batch(self, r):
        """

        :param r: [ num, frame, N ,D_in, D_in]
        :return:
        """
        angel = self.cal_roc_angel_batch(r)  # [ num, frame, N]
        sin = torch.sin(angel)
        mask = sin - 0.0 <= 1e-12
        sin += mask.float()
        log = (angel / (2 * sin)).unsqueeze(3).unsqueeze(4) * (r - r.permute(0, 1, 2, 4, 3))
        fi = torch.stack([log[:, :, :, 2, 1], log[:, :, :, 2, 0], log[:, :, :, 1, 0]], dim=3) #[ num, frame, N , 3]
        fi = fi / torch.norm(fi, 2, 3).unsqueeze(3) #[ num, frame, N , 3]
        return fi

    def cal_roc_angel(self, r):
        """
        Dismissed
        :param r:
        :return:
        """
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
        """
        Dismissed
        :param r:
        :return:
        """
        angel = self.cal_roc_angel(r)
        sin = torch.sin(angel)
        log = angel / (2 * sin) * (r - torch.t(r))
        # 下一行有点问题，与视觉slam， 对于反对称矩阵的fi1 2 3位置定义不一样
        fi = torch.tensor([log[2, 1], log[2, 0], log[1, 0]])
        fi = fi / torch.norm(fi)
        return fi


class Relu(torch.nn.Module):

    def __init__(self, device):
        super(Relu, self).__init__()
        self.device = device

    def forward(self, x):
        epslon = 0.3

        # batch version
        r_max = torch.max(x, torch.tensor(epslon).to(self.device))
        r_mask = x > 0
        r_positive = torch.mul(r_max, r_mask.float())
        r_min = torch.min(x, torch.tensor(-epslon).to(self.device))
        r_mask = x < 0
        r_negative = torch.mul(r_min, r_mask.float())
        Y = r_positive + r_negative
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
        # batch version
        x = x.permute(4, 0, 3, 1, 2)  # frame, N,  num, D_in, D_in
        w = self.w.permute(2, 0, 1)
        w = w.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x = torch.matmul(w, x)
        x = x.permute(1, 3, 4, 2, 0)

        return x


class BatchTrace(torch.nn.Module):

    def forward(self, x):
        assert x.shape[-1] == x.shape[-2]
        s = x.shape
        x = x.contiguous().view(-1, x.shape[-2], x.shape[-1])
        tr = x[:, 0, 0] + x[:, 1, 1] + x[:, 2, 2]
        tr = tr.view(s[:-2])
        return tr


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# net = LieNet()
# train = torch.randn(2, 3, 3, 342, 100).to(device)
# for i0 in range(2):
#     for i3 in range(342):
#         for i4 in range(100):
#             a = torch.randn(3, 3)
#             u, s, v = torch.svd(a)
#             train[i0, :, :, i3, i4] = u
# print(net.parameters())
# net.to(device)
# y = net(train)




