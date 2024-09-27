import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import R_Net
np.bool8 = np.bool

class REINFORCE:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,device):
        self.policy_net = R_Net.PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self,state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        # 创建一个类别分布
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self,transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        # 反向遍历
        for i in reversed(range(len(reward_list))):  
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            # 最后一个时间步开始反向遍历奖励列表，这样可以逐步累积reward
            G = self.gamma * G + reward
            # 每一步的损失函数
            loss = -log_prob * G
            loss.backward() 
        # 对每个参数做梯度下降
        self.optimizer.step()
