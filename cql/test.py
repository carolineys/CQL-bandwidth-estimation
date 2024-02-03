import torch_util as ptu
from replay_buffer import BWEReplayBuffer
from policies import TanhGaussianPolicy, MakeDeterministic
from trainer_cql import CQLTrainer
from networks import PolicyModel, BaseModel
from algorithm import TorchBatchRLAlgorithm
import numpy as np
import json
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import onnx
from onnx import helper
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import threading
import random
import pandas as pd


def replace_nan_with_zero(data):
    if isinstance(data, list):
        return [replace_nan_with_zero(item) for item in data]
    elif data == 'nan' or data == 'NaN' or data == 'NAN' or np.isnan(data):
        return 0
    else:
        return data

def get_file_paths(directory_path):
    file_list1 = os.listdir(directory_path)
    # file_list3 = os.listdir("../testbed_dataset/testing")[:500]
    print(datetime.now())

    file_paths = [
        os.path.join(directory_path, file) for file in file_list1 if file.endswith('.json')
    ]
    # file_paths += [
    #     os.path.join("../testbed_dataset/testing", file) for file in file_list3 if file.endswith('.json')
    # ]

    print(len(file_paths))
    random.shuffle(file_paths)
    return file_paths[:5000]

def compare_rl_and_heu(model):
    data_dir = "../emulated_dataset/testing/"
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)

    file_path = '../min_max.csv'
    data = pd.read_csv(file_path)
    obs_min = data['min'].to_numpy().astype(np.float32)
    obs_max = data['max'].to_numpy().astype(np.float32)

    total_avg_rel_diff1 = 0
    total_avg_rel_diff2 = 0
    num_files = 0

    for filename in data_files:
        base_filename = os.path.basename(filename)  # Get just the filename
        with open(filename, "r") as file:
            call_data = json.load(file)
        observations = np.array(replace_nan_with_zero(call_data["observations"]),  dtype=np.float32)
        observations = (observations - obs_min)/(obs_max - obs_min)
        bandwidth_predictions = np.asarray(replace_nan_with_zero(call_data['bandwidth_predictions']), dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)
        very_large_number = 1e10 
        true_capacity[np.isnan(true_capacity)] = very_large_number
        true_capacity[true_capacity == 0] = very_large_number

        observations = torch.tensor(observations, device='cuda:0')
        model.model.to('cuda:0')
        log_actions= model.model_forward(observations, deterministic=True)
        min_range=4.301
        max_range=6.9
        log_actions = log_actions[0].detach().cpu().numpy()
        scaled_action = ((log_actions + 1) / 2) * (max_range - min_range) + min_range
        actions = np.power(10,scaled_action).squeeze()

        avg_rel_diff1 = np.mean(np.abs(bandwidth_predictions - true_capacity) / true_capacity) * 100
        avg_rel_diff2 = np.mean(np.abs(np.power(10, scaled_action).squeeze() - true_capacity) / true_capacity) * 100

        # print(avg_rel_diff1, avg_rel_diff2)
        if np.isnan(avg_rel_diff2):
            continue
        else:
            total_avg_rel_diff1 += avg_rel_diff1
            total_avg_rel_diff2 += avg_rel_diff2
            num_files += 1

    mean_avg_rel_diff1_across_files = total_avg_rel_diff1 / num_files
    mean_avg_rel_diff2_across_files = total_avg_rel_diff2 / num_files


    print(mean_avg_rel_diff1_across_files, mean_avg_rel_diff2_across_files)



def eval_with_emulated_dataset(checkpoint_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_dim = 150
    action_dim = 1
    policy_model = PolicyModel(obs_dim, action_dim)
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    checkpoint_files.sort()  # Sort files if needed

    checkpoint_files = ["cp457.pth"]

    # Iterate over each checkpoint file
    for checkpoint_file in checkpoint_files:
        print(checkpoint_file)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        # Initialize the policy model
        saved_policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            model=policy_model,
            device=device
        )

        # Load the policy model
        saved_policy.load_policy_model(checkpoint_path)

        # Evaluate the model
        eval_result = compare_rl_and_heu(saved_policy)

def eval_onnx():
    data_dir = "../emulated_dataset/testing/"
    onnx_model = "../tmp/CQLModel.onnx"
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)
    ort_session = ort.InferenceSession(onnx_model)
    a_lst = []
    b_lst = []

    model = onnx.load(onnx_model)

    for filename in data_files:
        with open(filename, "r") as file:
            call_data = json.load(file)

        observations = np.asarray(call_data['observations'], dtype=np.float32)
        bandwidth_predictions = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)

        baseline_model_predictions = []
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)  
        for t in range(observations.shape[0]):  
            obs = observations[t:t+1,:].reshape(1,1,-1)
            feed_dict = {'obs': obs,
                        'hidden_states': hidden_state,
                        'cell_states': cell_state
                        }
            # print(obs.shape, hidden_state.shape, cell_state.shape)
            bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
            # print(bw_prediction.shape, hidden_state.shape, cell_state.shape)
            baseline_model_predictions.append(bw_prediction[0,0,0])
        baseline_model_predictions = np.asarray(baseline_model_predictions, dtype=np.float32)
        # print(baseline_model_predictions)
        # print(true_capacity)
        # print(baseline_model_predictions.shape)
        # print(true_capacity.shape)
        a = np.linalg.norm(baseline_model_predictions/1000000-true_capacity/1000000)
        b = np.linalg.norm(bandwidth_predictions/1000000-true_capacity/1000000)

        if not np.isnan(a) and not np.isnan(b):
            print(a,b)
            a_lst.append(a)
            b_lst.append(b)
    print("Final Results")
    print(np.mean(a), np.mean(b))


def enable_gpus(gpu_str='0'):
    if (gpu_str is not None):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enable_gpus()
    ptu.set_gpu_mode(True)

    # eval_with_emulated_dataset("./checkpoints_new/")
    eval_onnx()
