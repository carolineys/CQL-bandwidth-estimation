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
    return file_paths[:4000]

def experiment(device):
    obs_dim = 150
    action_dim = 1

    file_paths = get_file_paths('../testbed_dataset/training/')
    replay_buffer = BWEReplayBuffer(10000000, 150, 1, file_paths, device)

    qf1 = BaseModel(
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf1.to(device)
    qf2 = BaseModel(
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2.to(device)
    target_qf1 = BaseModel(
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    target_qf1.to(device)
    target_qf2 = BaseModel(
        input_size=obs_dim + action_dim,
        output_size=1
    )
    target_qf2.to(device)
    policy_model = PolicyModel(obs_dim, 2)
    policy_model.to(device)
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        model = policy_model,
        device=device
    )

    trainer = CQLTrainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        discount=0.99,
        soft_target_tau=5e-3,
        policy_lr=1E-4,
        qf_lr=3e-4,

        # soft_target_tau=5e-3,
        # policy_lr=1E-4,
        # qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
        num_qs=2,
        # CQL
        temp=1.0,
        min_q_version=3,
        min_q_weight=1.0,

        # lagrange
        with_lagrange=True,   # Defaults to true
        lagrange_thresh=10.0,
            
        # extra params
        num_random=10,
        max_q_backup=False,
        deterministic_backup=False,
        policy_eval_start=0,
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        replay_buffer=replay_buffer,
        batch_size=1024,
        max_path_length=5000,
        num_epochs=3000,
        num_trains_per_train_loop=5000,
        num_train_loops_per_epoch=1,
    )
    algorithm.to(device)
    algorithm.train()
    policy.save_model("./checkpoints_new/cp0.pth")
    return policy

def eval_policy(model):
    print(datetime.now())
    print("evaluation starts")
    data_dir = "../data"
    # data_dir = "../emulated_dataset/testing"
    figs_dir = "../figs"
    # onnx_model = "./onnx_models/Offline_RL_baseline_bandwidth_estimator_model.onnx"
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)[:20]

    file_path = '../min_max.csv'
    data = pd.read_csv(file_path)
    obs_min = data['min'].to_numpy().astype(np.float32)
    obs_max = data['max'].to_numpy().astype(np.float32)

    for filename in tqdm(data_files, desc="Processing"):
        base_filename = os.path.basename(filename)  # Get just the filename
        log_file_name = os.path.splitext(base_filename)[0] + ".log"
        log_file_path = os.path.join(figs_dir, log_file_name)
        with open(filename, "r") as file:
            call_data = json.load(file)
        observations = np.array(replace_nan_with_zero(call_data["observations"]),  dtype=np.float32)
        observations = (observations - obs_min)/(obs_max - obs_min)
        bandwidth_predictions = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)

        observations = torch.tensor(observations, device='cuda:0')
        model.model.to('cuda:0')
        log_actions= model.model_forward(observations, deterministic=True)
        min_range=4.301
        max_range=6.9
        log_actions = log_actions[0].detach().cpu().numpy()
        scaled_action = ((log_actions + 1) / 2) * (max_range - min_range) + min_range
        actions = np.power(10,scaled_action)
        model_predictions = actions

        with open(log_file_path, "w") as log_file:
            for model_pred, bw_pred in zip(model_predictions, bandwidth_predictions):
                log_file.write(f"{model_pred} {bw_pred}\n")

        fig = plt.figure(figsize=(8, 5))
        time_s = np.arange(0, observations.shape[0]*60,60)/1000
        
        plt.plot(time_s, model_predictions/1000, label='CQL RL', color='g')
        plt.plot(time_s, bandwidth_predictions/1000, label='BW Estimator '+call_data['policy_id'], color='r')
        plt.plot(time_s, true_capacity/1000, label='True Capacity', color='k')
        plt.ylabel("Bandwidth (Kbps)")
        plt.xlabel("Call Duration (second)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(figs_dir,os.path.basename(filename).replace(".json",".png")))
        plt.close()


def eval_onnx():
    data_dir = "../data"
    figs_dir = "../figs"
    onnx_model = "../tmp/CQLModel.onnx"
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)  
    ort_session = ort.InferenceSession(onnx_model)

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
          
            # for input in ort_session.get_inputs():
            #     print(f"Input Name: {input.name}")
            #     print(f"Input Shape: {input.shape}")
            #     print(f"Input Type: {input.type}")  
            # print("hihihih")
            feed_dict = {'obs': obs,
                        'hidden_states': hidden_state,
                        'cell_states': cell_state
                        }
            # print(obs.shape, hidden_state.shape, cell_state.shape)
            bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
            # print(bw_prediction.shape, hidden_state.shape, cell_state.shape)
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

def enable_gpus(gpu_str='0'):
    if (gpu_str is not None):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enable_gpus()
    ptu.set_gpu_mode(True)
    
    # policy = experiment(device)

    obs_dim = 150
    
    policy_model = PolicyModel(obs_dim, 2)
    saved_policy = TanhGaussianPolicy(
        obs_dim=150,
        action_dim=1,
        model= policy_model,
        device=device
    )
    saved_policy.load_policy_model("./cp700.pth")
    saved_policy.save_policy_onnx()
    eval_onnx()
    # eval_policy(saved_policy)
