#!/usr/bin/env python3
##
## EPITECH PROJECT, 2023
## Rainbow DQN
## File description:
## Rainbow Agent
##

import torch
import numpy as np
import random

import operator
BATCH_SIZE = 2
ALPHA = 0.5
BETA = 0.4
EPSILON = 10e-5

class ReplayBuffer:

    def __init__(self, N) -> None:
        self.transitions = dict()
        self.datas = dict()
        self.max_size = N
        self.curr_size = 0
        self.max_priority = 0.01
        self.sum_priorities = 0
        
    def init_transition(self, experience, priority, proba, weight) -> None:
        self.curr_size += 1
        index = self.curr_size % self.max_size
        self.transitions[index] = experience
        self.datas[index] = (priority, proba, weight, index)
        self.sum_priorities += priority

    def store_transition(self, experience) -> None:
        self.curr_size += 1
        index = self.curr_size % self.max_size

        if self.curr_size > self.max_size:
            data = random.choices(list(self.datas.items()), k=1)[0][1]
            self.sum_priorities -= data[0] ** ALPHA
            if data[0] == self.max_priority:
                self.priorities_max = max(self.datas.values(), key=operator.itemgetter(0))
            index = data[3]
            del self.datas[index]
            self.curr_size -= 1

        self.transitions[index] = experience
        priority = self.max_priority
        self.sum_priorities += priority ** ALPHA
        proba = (priority ** ALPHA) / (self.sum_priorities ** ALPHA)
        weight = ((1 / (self.curr_size * proba))) ** BETA
        self.datas[index] = (priority, proba, weight, index)
    
    def retrieve_transitions(self) -> tuple:
        datas = random.choices(self.datas, weights=[data[1] for data in self.datas.values()], k=BATCH_SIZE)

        transitions = [self.transitions[data[3]] for data in datas]
        indexes = [data[3] for data in datas]

        states = ([t[0] for t in transitions])
        actions = ([t[1] for t in transitions])
        rewards = ([t[2] for t in transitions])
        dones = ([t[3] for t in transitions])
        new_states = ([t[4] for t in transitions])

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1)
        new_states_t = torch.as_tensor(np.array(new_states), dtype=torch.float32)

        weights = torch.tensor([(1 / (self.curr_size * data[2])) ** BETA for data in datas])
        return states_t, actions_t, rewards_t, dones_t, new_states_t, weights, indexes

    def update_priority(self, new_priorities, indexes) -> None:
        for index, priority in zip(indexes, new_priorities):
            old_priority, proba, weight, index = self.datas[index]
            updated_priority = priority + EPSILON
            updated_proba = (1 / self.sum_priorities) * (updated_priority ** ALPHA)
            self.datas[index] = (updated_priority, updated_proba, weight, index)
            self.sum_priorities += updated_priority - old_priority


class RainbowAgent:

    def __init__(self, N) -> None:
        self.buffer = ReplayBuffer(N=N)

    def epsilon_greedy_policy(self, env, epsilon, greedy_action) -> int:
        bool_action = random.random() > epsilon
        if bool_action:
            action = greedy_action
        else:
            action = env.action_space.sample()
        return action
