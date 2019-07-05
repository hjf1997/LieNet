# implemented by junfeng Hu
import torch

def softmax_loss_LieNet(x, c):
    """

    :param X:
    :param c:
    :return:
    """
    # x = F.softmax(x, dim=1)
    # criterion = torch.nn.CrossEntropyLoss()
    # loss = criterion(x, c - 1)
    Xmax, _ = torch.max(x, dim=1)
    ex = torch.exp(x - Xmax.unsqueeze(1))
    t = Xmax + torch.log(torch.sum(ex, 1)) - x[[ i for i in range(x.shape[0])], (c-1).long()].view(-1)
    loss = torch.sum(t)
    return loss/x.shape[0]

# x = torch.randn(30, 20)
# y = torch.ones(30)
# print(softmax_loss_LieNet(x, y))