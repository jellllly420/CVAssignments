from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# ----------TODO------------
# Implement the PointNet 
# ----------TODO------------

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, d=1024):
        super(PointNetfeat, self).__init__()

        self.d = d
        self.global_feat = global_feat
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, self.d, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.d)
        self.relu = nn.ReLU()

    def forward(self, x):
        #print(x.shape)
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        vis_feature = torch.max(x, 1, keepdim = True)[0]
        #vis_feature = vis_feature(-1, )
        x = torch.max(x, 2)[0]

        if self.global_feat:
            return x, vis_feature
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, x.size()[1])
            return torch.cat([pointfeat, x], 1), vis_feature

class PointNetCls1024D(nn.Module):
    def __init__(self, k=2 ):
        super(PointNetCls1024D, self).__init__()

        self.k = k
        self.feat = PointNetfeat()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu =  nn.ReLU()

    def forward(self, x):
        x, vis_feature = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), vis_feature # vis_feature only for visualization, your can use other ways to obtain the vis_feature


class PointNetCls256D(nn.Module):
    def __init__(self, k=2 ):
        super(PointNetCls256D, self).__init__()

        self.k = k
        self.feat = PointNetfeat(d = 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.k)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu =  nn.ReLU()

    def forward(self, x):
        x, vis_feature = self.feat(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1), vis_feature





class PointNetSeg(nn.Module):
    def __init__(self, k = 2):
        super(PointNetSeg, self).__init__()

        self.k = k
        self.feat = PointNetfeat(global_feat = False)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu =  nn.ReLU()


    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        #print(x.shape)
        x, _ = self.feat(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = F.log_softmax(x.permute(0, 2, 1), dim = 2)
        #print(x.shape)

        return x

