from collections import OrderedDict

import numpy as np
import torch
from datetime import datetime
import json
import abc
import csv
import os
import pandas as pd

    
def replace_nan_with_zero(data):
    if isinstance(data, list):
        return [replace_nan_with_zero(item) for item in data]
    elif data == 'nan' or data == 'NaN' or data == 'NAN' or np.isnan(data):
        return 0
    else:
        return data

class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return


class BWEReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        file_paths,
        device
    ):
        self.file_paths = file_paths
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        self._top = 0
        self._size = 0
        self.device = device
        self._obs_min = None
        self._obs_max = None

    def clear_buffer(self):
        self._size = 0
        self._top = 0
        self._observations = np.zeros((self._max_replay_buffer_size, self._observation_dim))
        self._next_obs = np.zeros((self._max_replay_buffer_size, self._observation_dim))
        self._actions = np.zeros((self._max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros((self._max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((self._max_replay_buffer_size, 1), dtype='uint8')


    def add_sample(self, observation, action, reward, next_observation,
                   terminal):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        return batch


    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    
    def add_path(self, path):
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
            )
        self.terminate_episode()

    
    def read_stats(self):
        file_path = '../min_max.csv'
        data = pd.read_csv(file_path)
        self._obs_min = data['min'].to_numpy()
        self._obs_max = data['max'].to_numpy()
        # shape is (150,)


    def load_to_buffer(self, file_list):
        print("start loading " + str(len(file_list)) + " to buffer "+ str(datetime.now()))

        # Loop through each file in the directory
        cnt = 0
        for file_path in file_list:
            cnt += 1
            # if cnt % 500 == 0:
            #     print(datetime.now())
            #     print("loading files: " + str(cnt) + "/" + str(len(file_list)))
            # Check if the file has a JSON extension
            if file_path.endswith('.json'):

                # Open the JSON file and load its content into a dictionary
                with open(file_path, 'r') as json_file:
                    data_dictionary = json.load(json_file)
                    new_dict = {}
                    observations = np.array(replace_nan_with_zero(data_dictionary["observations"]))
                    observations = (observations - self._obs_min)/(self._obs_max - self._obs_min)
                  
                    new_dict["observations"] = observations # shape is (nx150)
                    next_observations = observations[1:,:]
                    next_observations = (next_observations - self._obs_min)/(self._obs_max - self._obs_min)
                    zeros_array = np.zeros((1, 150))
                    next_observations = np.vstack((next_observations, zeros_array))
                    new_dict["next_observations"] = next_observations

                    terminals = np.zeros(len(next_observations), dtype=int)
                    terminals[-1] = 1
                    new_dict["terminals"] = terminals

                    new_dict["rewards"] = np.array(replace_nan_with_zero(data_dictionary["video_quality"]))
                    actions = np.array(replace_nan_with_zero(data_dictionary["bandwidth_predictions"]))
                    epsilon = 1e-10  # Small positive number
                    log_actions = np.log10(actions + epsilon)
                    min_range=4.301
                    max_range=6.9
                    scaled_actions = -1 + 2 * (log_actions - min_range)/(max_range - min_range)
                    new_dict["actions"] = scaled_actions

                    # for key, value in new_dict.items():
                    #     print(key, value.shape)
                
                    self.add_path(new_dict)
        diagnose = self.get_diagnostics()
        print(diagnose)
