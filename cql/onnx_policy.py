import numpy as np
import torch
from torch import nn as nn
import os
from tqdm import tqdm
import onnxruntime as ort

from torch_core import eval_np
from torch_distributions import TanhNormal
from networks import Mlp, CustomModel
from base_policy import Policy, ExplorationPolicy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0

def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5*torch.log(one_plus_x/ one_minus_x)


class ONNXPolicy(Mlp, ExplorationPolicy):
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
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            device='cpu',
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )

        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim).to(device)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def log_prob(self, obs, actions):
        raw_actions = atanh(actions)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob.sum(-1)

    def forward(
            self,
            obs, hidden_states, cell_states,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        obs = obs.detach()
        hidden_states = hidden_states.detach()
        cell_states = cell_states.detach()
        h = obs
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if i == 0 and len(self.layer_batch_norms) >= i:
                h = self.layer_batch_norms[i](h)
            h = self.hidden_activation(h)
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

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
        action = action * (11000000 - 20000) + 20000

        mean = mean * (11000000 - 20000) + 20000

        N = obs.shape
        print("obs" + str(obs.shape))
        # state_out: float32[1, N], N is the same as for the input hidden states dimension
        # cell_out: float32[1, N], N is the same as for the input cell states dimension
        state_out = np.zeros((1, N), dtype=np.float32)
        cell_out = np.zeros((1, N), dtype=np.float32)

        return mean, std, state_out, cell_out

    def save_model(self, filepath):
        """
        Save the model to a file.
        """
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        """
        Load the model from a file.
        """
        self.load_state_dict(torch.load(filepath))

    def save_onnx_model(self):
        self.eval()
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()

        model_path = "../tmp/onnxCQLModel.onnx"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.to("cpu")

        BS = 1
        T = 2000
        obs_dim = 150
        hidden_size = 1

        # Create dummy inputs without gradients
        dummy_inputs = np.random.uniform(0, 1, size=(BS, T, obs_dim)).astype(np.float32)
        torch_dummy_inputs = torch.from_numpy(dummy_inputs).requires_grad_(False)
        torch_initial_hidden_state = torch.zeros((BS, hidden_size), requires_grad=False)
        torch_initial_cell_state = torch.zeros((BS, hidden_size), requires_grad=False)

        torch.onnx.export(
            self,
            (torch_dummy_inputs, torch_initial_hidden_state, torch_initial_cell_state),
            model_path,
            opset_version=11,
            input_names=['obs', 'hidden_states', 'cell_states'], # the model's input names
            output_names=['output', 'state_out', 'cell_out'], # the model's output names
            dynamic_axes={'obs': {0: 'batch_size', 1: 'sequence'}, # dynamic batch size and sequence length
                        'hidden_states': {0: 'batch_size'},
                        'cell_states': {0: 'batch_size'},
                        'output': {0: 'batch_size'},
                        'state_out': {0: 'batch_size'},
                        'cell_out': {0: 'batch_size'}}
        )

def predict():
    data_dir = "../data"
    figs_dir = "../figs"
    onnx_model = "../tmp/onnxCQLModel.onnx"
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)  
    ort_session = ort.InferenceSession(onnx_model)

    model = onnx.load(onnx_model)
    print("finish load")

    # Iterate through the graph nodes
    for i, node in enumerate(model.graph.node):
        print(f"Node {i + 1}:")
        print(f"  Name: {node.name}")
        print(f"  OpType: {node.op_type}")
        print(f"  Input(s): {node.input}")
        print(f"  Output(s): {node.output}")
        print()


    for filename in tqdm(data_files, desc="Processing"):
        with open(filename, "r") as file:
            call_data = json.load(file)

        observations = np.asarray(call_data['observations'], dtype=np.float32)
        bandwidth_predictions = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)

        baseline_model_predictions = []
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)    
        for t in range(observations.shape[0]):
            feed_dict = {'obs': observations[t:t+1,:].reshape(1,1,-1),
                        'hidden_states': hidden_state,
                        'cell_states': cell_state
                        }
            bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
            baseline_model_predictions.append(bw_prediction[0,0,0])
        baseline_model_predictions = np.asarray(baseline_model_predictions, dtype=np.float32)
        fig = plt.figure(figsize=(8, 5))
        time_s = np.arange(0, observations.shape[0]*60,60)/1000
        plt.plot(time_s, baseline_model_predictions/1000, label='Offline RL Baseline', color='g')
        plt.plot(time_s, bandwidth_predictions/1000, label='BW Estimator '+call_data['policy_id'], color='r')
        plt.plot(time_s, true_capacity/1000, label='True Capacity', color='k')
        plt.ylabel("Bandwidth (Kbps)")
        plt.xlabel("Call Duration (second)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(figs_dir,os.path.basename(filename).replace(".json",".png")))
        plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = 128
    policy = ONNXPolicy(
        obs_dim=150,
        action_dim=1,
        hidden_sizes=[M, M, M, 64, 32],
        device=device
    )
    with torch.no_grad():
        policy.load_model("./checkpoints/cp1.pth")
    print("before save")
    policy.save_onnx_model()
    print("after save")
    predict()