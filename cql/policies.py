import numpy as np
import torch
from torch import nn as nn

from torch_core import eval_np
from torch_distributions import TanhNormal
from networks import BaseModel, PolicyModel
from base_policy import Policy, ExplorationPolicy
import os
import pandas as pd

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0

def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5*torch.log(one_plus_x/ one_minus_x)

class ExtendedModel(PolicyModel):
    def __init__(self, input_size, output_size, init_w=1e-3):
        super(ExtendedModel, self).__init__(input_size, output_size, init_w)
        self.obs_min = None
        self.obs_max = None

    def load_min_max(self):
        file_path = '../min_max.csv'
        data = pd.read_csv(file_path)
        self.obs_min = torch.from_numpy(data['min'].to_numpy().astype(np.float32)).view(1, 1, -1)
        self.obs_max = torch.from_numpy(data['max'].to_numpy().astype(np.float32)).view(1, 1, -1)
        print(self.obs_min.shape, self.obs_max.shape)


    def forward(self, obs, hidden_states=0, cell_states=0):
        # Use the parent class's forward method
        print("hahah", obs.shape, self.obs_min.shape, self.obs_max.shape)
        x = (obs - self.obs_min)/(self.obs_max - self.obs_min)
        # x = obs
        print(x.shape)
    
        outputs, h, c = super().forward(x, hidden_states, cell_states)
        print(outputs.shape)
        mean, log_std = torch.split(outputs, 1, dim=2)
        log_action = torch.tanh(mean)
        min_range=4.301
        max_range=6.9
        scaled_action = ((log_action + 1) / 2) * (max_range - min_range) + min_range
        action = torch.pow(10,scaled_action)
        outputs = torch.cat([action, log_std], dim=2)
        return outputs, hidden_states, cell_states

class TanhGaussianPolicy(ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            obs_dim,
            action_dim,
            model,
            std=None,
            init_w=1e-3,
            device='cpu',
            **kwargs
    ):
        super().__init__()

        self.model = model
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.log_std = None
        self.std = std
        # if std is None:
        #     last_hidden_size = obs_dim
        #     if len(hidden_sizes) > 0:
        #         last_hidden_size = hidden_sizes[-1]
        #     self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim).to(device)
        #     self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        #     self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        # else:
        #     self.log_std = np.log(std)
        #     assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def log_prob(self, obs, actions):
        raw_actions = atanh(actions)
        
        h = obs.unsqueeze(1)

        outputs, h, c = self.model(h)
        outputs = outputs.squeeze(1)
        mean, log_std = torch.split(outputs, self.action_dim, dim=1)
        std = torch.exp(log_std)
        # for i, fc in enumerate(self.fcs):
        #     h = self.hidden_activation(fc(h))
        # mean = self.last_fc(h)
        # mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        # if self.std is None:
        #     log_std = self.last_fc_log_std(h)
        #     log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        #     std = torch.exp(log_std)
        # else:
        #     std = self.std
        #     log_std = self.log_std

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob.sum(-1)

    def model_forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        # h = obs
        # for i, fc in enumerate(self.fcs):
        #     h = fc(h)
        #     if i == 0 and len(self.layer_batch_norms) >= i:
        #         h = self.layer_batch_norms[i](h)
        #     h = self.hidden_activation(h)
        # mean = self.last_fc(h)
        # if self.std is None:
        #     log_std = self.last_fc_log_std(h)
        #     log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        #     std = torch.exp(log_std)
        # else:
        #     std = self.std
        #     log_std = self.log_std
        h = obs.unsqueeze(1)

        outputs, h, c = self.model(h)

        outputs = outputs.squeeze(1)
        mean, log_std = torch.split(outputs, self.action_dim, dim=1)
        std = torch.exp(log_std)

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

    def save_policy_model(self, filepath):
        """
        Save the model to a file.
        """
        # print("Model's state_dict:")
        # for param_tensor in self.state_dict():
        #     print(param_tensor, "\t", self.state_dict()[param_tensor].size())
        torch.save(self.model.state_dict(), filepath)

    def load_policy_model(self, filepath):
        """
        Load the model from a file.
        """
        self.model.load_state_dict(torch.load(filepath))

    def save_policy_onnx(self):
        extended_model = ExtendedModel(150,1)
        extended_model.load_min_max()
        extended_model.load_state_dict(self.model.state_dict())
        extended_model.eval()

        model_path = "../tmp/CQLModel.onnx"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        extended_model.to("cpu")

        BS = 1
        T = 1
        obs_dim = 150
        hidden_size = 1

        # Create dummy inputs without gradients
        dummy_inputs = np.random.uniform(0, 1, size=(BS, T, obs_dim)).astype(np.float32)
        torch_dummy_inputs = torch.from_numpy(dummy_inputs).requires_grad_(False)
        torch_initial_hidden_state = torch.zeros((BS, hidden_size), requires_grad=False)
        torch_initial_cell_state = torch.zeros((BS, hidden_size), requires_grad=False)

        torch.onnx.export(
            extended_model,
            (torch_dummy_inputs, torch_initial_hidden_state, torch_initial_cell_state),
            model_path,
            opset_version=11,
            input_names=['obs', 'hidden_states', 'cell_states'],
            output_names=['output', 'state_out', 'cell_out'],
            # dynamic_axes={'obs': {0: 'batch_size', 1: 'sequence'}, # dynamic batch size and sequence length
            #             'hidden_states': {0: 'batch_size'},
            #             'cell_states': {0: 'batch_size'},
            #             'output': {0: 'batch_size'},
            #             'state_out': {0: 'batch_size'},
            #             'cell_out': {0: 'batch_size'}}
        )


class MakeDeterministic(nn.Module, Policy):
    def __init__(self, stochastic_policy):
        super().__init__()
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)
