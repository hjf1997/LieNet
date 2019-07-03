import torch
import numpy as np
import torch.nn.functional as F

def softmax_loss_LieNet(x, c):
    """

    :param X:
    :param c:
    :return:
    """
    x = F.softmax(x, dim=1)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(x, c - 1)
    return loss

