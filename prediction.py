# implemented by junfeng Hu
import sys
sys.path.append('./code')

import torch
from LieNet import LieNet
from torch.utils.data import DataLoader
from load_g3d import G3dDataset
from utils import egrad2rgrad, retr
from torch import autograd
import torch.nn.functional as F


def acc(pred, labels):
    pred = F.softmax(pred, 1)
    _, pred = torch.max(pred, 1)
    train_correct = (pred == (labels - 1)).sum().float()
    acc = train_correct / labels.shape[0]
    return acc


def train(datat, iter, train):

    if datat == 'g3d':
        train_dataset = G3dDataset(train=True)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        test_dataset = G3dDataset(train=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0)
    elif datat == 'CSL':
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    critetion = torch.nn.CrossEntropyLoss()
    net = LieNet(device)
    net.to(device)

    if torch.cuda.device_count() > 1:
        print("Let.s use {} GPUs!".format(str(torch.cuda.device_count())))
        net = torch.nn.DataParallel(net)

    if train:

        for epoch in range(iter):
            running_loss = 0.0
            accuracy = 0.0
            for i, data in enumerate(train_loader, 0):

                inputs = data['fea'].float().to(device)
                labels = data['label'].to(device)
                with autograd.detect_anomaly():
                    outputs = net(inputs)
                    loss_ = critetion(outputs, labels-1)
                    running_loss += loss_
                    accuracy += acc(outputs, labels)
                    net.zero_grad()
                    loss_.backward()

                _ = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                count = 0
                for param in net.parameters():
                    if count < 3:
                        count += 1
                        #grad = egrad2rgrad(param.data.permute(1, 0, 2),
                        #                   param.grad.data.permute(1, 0, 2)).permute(1, 0, 2)
                        #grad = retr(param.data.permute(1,0,2), -0.01*grad.permute(1,0,2),
                        #            param.data.shape[-1]).permute(1,0,2)
                        grad = egrad2rgrad(param.data, param.grad.data)
                        grad = retr(param.data, -0.01*grad, param.data.shape[-1])
                        param.data.sub_(param.data)
                        param.data.sub_(grad * -1)
                    else:
                        count += 1
                        param.data.sub_(0.01 * param.grad.data)

            print("epoch:{}, train_loss:{}".format(str(epoch+1), str((running_loss/(i+1)).item())))
            print("Train accuracy:{}".format(str((accuracy/(i+1)).item())))

            accuracy = 0.0
            running_loss = 0.0

            with torch.no_grad():
                for i, data in enumerate(test_loader, 0):

                    inputs = data['fea'].float().to(device)
                    labels = data['label'].to(device)
                    outputs = net(inputs)
                    loss_ = critetion(outputs, labels-1)
                    running_loss += loss_
                    accuracy += acc(outputs, labels)
            print("epoch:{}, test_loss:{}".format(str(epoch + 1), str((running_loss / (i + 1)).item())))
            print("Test accuracy:{}".format(str((accuracy / (i + 1)).item())))
            print("-------------------------------------------------------------------")

            if epoch % 1000 == 0:
                torch.save(net.state_dict(), './model/{}.pth'.format(str(epoch)))


train('g3d', 4000, True)




