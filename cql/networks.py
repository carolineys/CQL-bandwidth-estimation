"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from base_policy import Policy
import torch_util as ptu
from torch_core import eval_np
from normalizer import TorchFixedNormalizer

"""
Contain some self-contained modules.
"""
import torch
import torch.nn as nn


LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0


class HuberLoss(nn.Module):
    def __init__(self, delta=1):
        super().__init__()
        self.huber_loss_delta1 = nn.SmoothL1Loss()
        self.delta = delta

    def forward(self, x, x_hat):
        loss = self.huber_loss_delta1(x / self.delta, x_hat / self.delta)
        return loss * self.delta * self.delta


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output



def identity(x):
    return x


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=F.tanh,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            batch_norm=1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        self.layer_batch_norms = []
        in_size = input_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(batch_norm):
            self.layer_batch_norms.append(nn.BatchNorm1d(hidden_sizes[0]).to(device))
 
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size).to(device)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size).to(device)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


# input_size = 1000  # Adjust based on your actual input size
# hidden_sizes = [256, 128, 64]  # Adjust based on your architecture
# output_size = 10  # Adjust based on your output size
# model = CustomModel(input_size, hidden_sizes, output_size)


class BaseModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseModel, self).__init__()
        
        # Define layers
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.last_fc = nn.Linear(32, output_size)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.last_fc(x)
        return x

# class PolicyModel(nn.Module):
#     def __init__(self, input_size, output_size, init_w=1e-3):
#         super(PolicyModel, self).__init__()
        
#         # Define layers
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 64)
#         self.fc5 = nn.Linear(64, 32)
#         self.last_fc_log_std = nn.Linear(32, 1)
#         self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
#         self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
#         self.last_fc_mean = nn.Linear(32, 1)

#     def forward(self, x, h=0, c=0):
#         # Define forward pass
#         x = x.squeeze(1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.relu(x)
#         x = self.fc4(x)
#         x = self.relu(x)
#         x = self.fc5(x)
#         x = self.relu(x)
#         mean = self.last_fc_mean(x)
#         log_std = self.last_fc_log_std(x)
#         mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
#         log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
#         x = torch.cat((mean, log_std), dim=1)
#         x = x.unsqueeze(1)
#         return x, h, c

class PolicyModel(nn.Module):
    def __init__(self, input_size, output_size, init_w=1e-3):
        super(PolicyModel, self).__init__()
        
        # Define layers
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.last_fc_log_std = nn.Linear(32, 1)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        self.last_fc_mean = nn.Linear(32, 1)

    def forward(self, x, h=0, c=0):
        # Define forward pass
        x = x.squeeze(1)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        mean = self.last_fc_mean(x)
        log_std = self.last_fc_log_std(x)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        x = torch.cat((mean, log_std), dim=1)
        x = x.unsqueeze(1)
        return x, h, c

class MlBandwidthEstimator(nn.Module):
    def __init__(self, in_feat, hidden_size=256):
        super().__init__()
        self.in_feat = in_feat
        self.hidden_size = hidden_size
        # In this example, an lstm is used to construct a stateful deep net
        self.lstm = nn.LSTM(in_feat, hidden_size, num_layers=1, batch_first=True)
        # output layer: mean and standard deviation of bandwidth estimates
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, h, c):
        h, c = (h.unsqueeze(0), c.unsqueeze(0)) # adding layer dimension
        self._features, [h, c] = self.lstm(x, [h, c])
        x = self.fc(self._features)
        h, c = (h.squeeze(0), c.squeeze(0)) # removing layer dimension
        return x, h, c  