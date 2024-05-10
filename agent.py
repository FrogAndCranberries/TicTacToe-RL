import environment
import numpy as np
import torch as t
import tqdm

# Setup
device = "cpu"

class DQN_Agent:

    def __init__(self, board_size:int, streak:int, hidden_channels:list = [8, 16]) -> None:
        self.obs_size = board_size
        self.streak = streak
        kernel_size = streak
        linear_in_features = hidden_channels[1] * ((board_size - (kernel_size - 1) * 2) ** 2)
        out_features = 2

        self.value_network = t.nn.Sequential(
            t.nn.Conv2d(in_channels=3,out_channels=hidden_channels[0],kernel_size=kernel_size,stride=1,padding="valid"),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=hidden_channels[0],out_channels=hidden_channels[1],kernel_size=kernel_size,stride=1,padding="valid"),
            t.nn.ReLU(),
            t.nn.Flatten(),
            t.nn.Linear(in_features=linear_in_features, out_features=2),
            t.nn.Sigmoid()
        )

        self.target_network = t.nn.Sequential(
            t.nn.Conv2d(in_channels=3,out_channels=hidden_channels[0],kernel_size=kernel_size,stride=1,padding="valid"),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=hidden_channels[0],out_channels=hidden_channels[1],kernel_size=kernel_size,stride=1,padding="valid"),
            t.nn.ReLU(),
            t.nn.Flatten(),
            t.nn.Linear(in_features=linear_in_features, out_features=2),
            t.nn.Sigmoid()
        )

        