import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import pdb

from utils.th_utils import orthogonal_init_


class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args, mean=0, std=0.1):
        super(NRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        torch.manual_seed(self.args.seed)
        if args.use_n_lambda:
            self.fc2 = nn.Linear(args.hidden_dim, args.n_lambda * args.n_actions)
        else:
            self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        # TODO:初始化形式改为了gaussian
        nn.init.normal_(self.fc1.weight, mean=mean, std=std)
        nn.init.normal_(self.fc2.weight, mean=mean, std=std)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
    def random_init_hidden(self, seed):
        torch.manual_seed(seed)
        # 创建权重张量
        weights = torch.empty(1, self.hidden_dim)
        # 使用nn.init.normal_初始化权重张量，均值为0，标准差为0.01
        nn.init.normal_(weights, mean=0, std=0.01)
        # 确保权重张量的requires_grad为True，以便在反向传播时进行优化
        weights.requires_grad = True
        return weights
        # return nn.init.normal(self.fc1.weight.new(1, self.args.hidden_dim), mean=0.0, std=0.01)

    def forward(self, inputs, hidden_state):

        # Batch x Agents x Dim_state
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        hh = self.rnn(x, h_in)

        q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)