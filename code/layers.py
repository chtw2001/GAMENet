import torch
import math
import torch.nn as nn

from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        # gcn1 -> med_voc, 64
        # gcn2 -> 64, 64
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # gcn1의 self.x는 단순히 초기에 곱할 것이 없기 때문에 대각행렬로 곱하는 것 인듯
        
        # gcn1 -> (med_voc, med_voc) * (med_voc, 64) -> (med_voc, 64)
        # gcn2 -> (med_voc, 64) * (64, 64) -> (med_voc, 64)
        support = torch.mm(input, self.weight)
        # gcn1 -> (med_voc, med_voc) * (med_voc, 64) -> (med_voc, 64)
        # gcn2 -> (med_voc, med_voc) * (med_voc, 64) -> (med_voc, 64)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'