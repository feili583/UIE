import torch
import torch_model_utils as tmu

from torch import nn
from torch.distributions.utils import lazy_property


class TreeCRFVanilla(nn.Module):

    def __init__(self, log_potentials, lengths=None):
        self.log_potentials = log_potentials
        self.lengths = lengths
        return

    @lazy_property
    def entropy(self):
        batch_size = self.log_potentials.size(0)
        device = self.log_potentials.device
        return torch.zeros(batch_size).to(device)

    @lazy_property
    def partition(self):
        # Inside algorithm
        device = self.log_potentials.device
        batch_size = self.log_potentials.size(0)
        max_len = self.log_potentials.size(1)
        label_size = self.log_potentials.size(3)

        #每个字对应标签的logits，不同位置为0，自底向上,先计算单个节点的，在此基础上再计算两个节点的，再计算三个节点的，其中1，3计算方式
        #1，3=1，2+2，3
        beta = torch.zeros_like(self.log_potentials).to(device)
        #span=0时的分数
        for i in range(max_len):
            beta[:, i, i] = self.log_potentials[:, i, i]
        torch.set_printoptions(profile="full")
        # print('q',beta)
        #span长度
        for d in range(1, max_len):
            #起始位置
            for i in range(max_len - d):
                #末尾位置
                j = i + d
                #batch_size,1,d,label_size->batch_size, d, label_size, 1,相当于i:j这些列，每行对应相应的标签的分数
                before_lse_1 = beta[:, i, i:j].view(batch_size, d, label_size, 1)
                # print('q',before_lse_1.shape)
                #batch_size,d,1,label_size->batch_size, d, 1, label_size，相当于i+1:j+1这些行，每列对应相应的标签的分数
                before_lse_2 = beta[:, i + 1: j + 1, j].view(batch_size, d, 1, label_size)
                # print('q',before_lse_2.shape)
                #batch_size, d, label_size, 1+batch_size, d, 1, label_size->batch_size,d,label_size,label_size
                #计算从i到j的所有span的分数
                before_lse = (before_lse_1 + before_lse_2).reshape(batch_size, -1)
                # print('q',before_lse.shape)
                after_lse = torch.logsumexp(before_lse, -1).view(batch_size, 1)
                # print('q',after_lse)
                beta[:, i, j] = self.log_potentials[:, i, j] + after_lse
                # print('q',before_lse_1)
                # print('q',before_lse_2)
                # print('q',before_lse)
                # print('q',after_lse)
                # print('q',beta)
                # stop
        if (self.lengths is None):
            before_lse = beta[:, 0, max_len - 1]
        else:
            before_lse = tmu.batch_index_select(beta[:, 0], self.lengths - 1)
        log_z = torch.logsumexp(before_lse, -1)
        return log_z

    @lazy_property
    def argmax(self):
        raise NotImplementedError('slow argmax not implemented!')
        return
