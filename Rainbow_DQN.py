#!/usr/bin/env python3
##
## EPITECH PROJECT, 2023
## Rainbow DQN
## File description:
## Rainbow DQN main file
##

# This is all the import needed:
import numpy as np
import math

import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

from Neural_Network import *
from Rainbow_Agent import *

# Constants:
EPISODES = 500
FRAME = 1000

GAMMA = 0.99
LR=5e-4
MEMORY_CAPACITY = 10
EPS_START = 1.0 # Epsilon parameters
EPS_END = 0.05
EPS_DECAY = 5e-4
UPDATE_FREQUENCY = 1000

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load environment:
env = gym.make("LunarLander-v2", render_mode="rgb_array")
# Load functions and classes needed:
Rainbow = RainbowAgent(N=MEMORY_CAPACITY)
online_network = NeuralNetwork(env=env).to(device=device)
target_network = NeuralNetwork(env=env).to(device=device)

optimizer = torch.optim.RMSprop(online_network.parameters(), lr=LR)
loss_fn = torch.nn.SmoothL1Loss(reduction='none')

state, _ = env.reset()
for frame in tqdm(range(MEMORY_CAPACITY)):
    action = env.action_space.sample()

    new_state, reward, done, _, _ = env.step(action)

    experience = (state, action, reward, done, new_state)
    Rainbow.buffer.init_transition(experience=experience, priority=10e-5, proba=(1 / MEMORY_CAPACITY), weight=1.0)

    state = new_state

    if done:
        state, _ = env.reset()

# Epsilon
epsilon = 1.0
# Steps
steps = 0

# Training :
for episode in tqdm(range(1, EPISODES + 1)):
    all_rewards = []
    state, info = env.reset()
    BETA = 0.4 + (1.0 - 0.4) * (episode / EPISODES)
    # epsilon = EPS_START - (EPS_START - EPS_END) * (episode / EPISODES)
    # epsilon = max(0.1, epsilon - 1.0 / EPISODES * 2)
    # epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-EPS_DECAY * episode)

    for frame in range(1, FRAME + 1):
        epsilon = max(0.1, epsilon * 0.99995)

        # Taking an action
        action = get_action(state=state, env=env, epsilon=epsilon, Agent=Rainbow, online_network=online_network)

        # Observe env return values
        new_state, reward, termination, truncation, _ = env.step(action)
        all_rewards.append(reward)

        # Add the experience to the experience relay
        Rainbow.buffer.store_transition((state, action, reward, termination, new_state))

        # Go to next state
        state = new_state
        # Sample an experience and get its state, action, reward and new_state
        state_t, action_t, reward_t, termination_t, new_state_t, weights, indexes = Rainbow.buffer.retrieve_transitions()

        # Implement loss
        max_next_state = target_network.forward(new_state_t).max(dim=1, keepdim=True)[0]
        y = reward_t + (GAMMA * max_next_state * (1 - termination_t))

        current_q = online_network.forward(state_t).gather(dim=1, index=action_t)

        error = loss_fn(y, current_q)
        loss = torch.mean(error * weights)
        # Reset gradient
        optimizer.zero_grad()
        # Backpropagate
        loss.backward()
        # Apply gradient on the Neural Network
        optimizer.step()

        new_priorities = error ** ALPHA
        Rainbow.buffer.update_priority(new_priorities=new_priorities, indexes=indexes)

        if steps % UPDATE_FREQUENCY == 0:
            copy_nn_parameters(target_network, online_network)

        # If it is the end pass to the next episode
        if termination or truncation:
            break

    if episode % 100 == 0:
        print("We are at episode %d\tmean reward is %lf and sum reward is %lf\tepsilon is %lf" % (episode, np.mean(all_rewards), sum(all_rewards), epsilon))
        if sum(all_rewards) >= 200:
            break

env.close()

# Compute the real deal...
env = gym.make("LunarLander-v2", render_mode="human")

# online_network.load_state_dict(torch.load("Nasa.pt"))
# online_network.eval()

epsilon = 0.0
for episode in range(1, 6):
    rewards = []
    state, _ = env.reset()
    for k in range(1, 1000):
        # Taking an action
        action = get_action(state=state, env=env, epsilon=epsilon, Agent=Rainbow, online_network=online_network)

        # Observe env return values
        new_state, reward, termination, truncation, _ = env.step(action)

        # Go to the next state for next action
        state = new_state

        # Get all the rewards
        rewards.append(reward)

        # If it is the end pass to the next episode
        if termination or truncation:
            break

    # Print the mean recent reward every 50 episodes
    if episode % 5 == 0:
        print(f"Episode {episode:>6}: \tR:{np.mean(rewards):>6.3f}")

env.close()
