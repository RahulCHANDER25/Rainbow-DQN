#!/usr/bin/env python3
##
## EPITECH PROJECT, 2023
## Rainbow DQN
## File description:
## Neural Network for the Rainbow Agent
##

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import gymnasium as gym

from Rainbow_Agent import RainbowAgent

class NeuralNetwork(nn.Module):

    def __init__(self, env) -> None:
        super().__init__()

        self.epsilon = 1.0
        self.state_nn = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_nn = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )

    def forward(self, x):
        state_value = self.state_nn(x)
        advantage = self.advantage_nn(x)
        average_advantage = advantage.mean()
        return state_value + (advantage - average_advantage)

def get_action(state: np.ndarray, env: gym.wrappers.time_limit.TimeLimit, epsilon: float, Agent: RainbowAgent, online_network: NeuralNetwork) -> int:
    states = torch.as_tensor(state, dtype=torch.float32)
    q_values = online_network.forward(states)
    greedy_action = q_values.argmax().detach().item()
    action = Agent.epsilon_greedy_policy(env=env, epsilon=epsilon, greedy_action=greedy_action)
    return action


def gradient_descent(loss, optimizer) -> None:
    # Reset gradient
    optimizer.zero_grad()
    # Backpropagate
    loss.backward()
    # Apply gradient on the Neural Network
    optimizer.step()


def copy_nn_parameters(target_network: NeuralNetwork, online_network: NeuralNetwork) -> None:
    target_network.load_state_dict(online_network.state_dict())
