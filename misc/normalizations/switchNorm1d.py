import torch
import torch.nn as nn
import torch.nn.functional as F


class switch_norm1d(nn.Module):
    def __init__(self, num_features, num_time_steps, eps=1e-5, momentum=0.1):
        super(switch_norm1d, self).__init__()
        self.b_norm = nn.BatchNorm1d(
            num_features=num_features, eps=eps, momentum=momentum)
        self.i_norm = nn.InstanceNorm1d(
            num_features=num_features, eps=eps, momentum=momentum)
        self.l_norm = nn.LayerNorm(
            normalized_shape=[num_features, num_time_steps])

        self.weight = nn.Parameter(torch.ones(3))   # noqa

    def forward(self, x):
        w = F.softmax(self.weight, 0)
        return (w[0] * self.b_norm(x)) + (w[1] * self.i_norm(x)) + (w[2] * self.l_norm(x))
