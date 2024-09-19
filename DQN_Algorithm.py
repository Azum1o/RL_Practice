import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import DQN_Net
np.bool8 = np.bool


class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device):
        self.action_dim = action_dim
        # 当前网络
        self.q_net = DQN_Net.Qnet(state_dim,hidden_dim,self.action_dim).to(device)
        # target网络
        self.taget_q_net = DQN_Net.Qnet(state_dim,hidden_dim,self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        # 折扣因子
        self.gamma = gamma
        # ε-Greedy策略
        self.epsilon = epsilon
        # targer网络更新频率
        self.taget_update = target_update
        # 记录更新次数
        self.count = 0
        # 设备选择
        self.device = device

        # 根据ε-Greedy策略选择动作
    def take_action(self,state):
        if np.random.random() < self.epsilon:
            # 生成一个[0,action_dim-1]的随机整数
            action = np.random.randint(self.action_dim)
        else:
            # state变为一个形状为(1, 4)的PyTorch张量，代表一个状态下包含的四种信息
            state = torch.tensor([state],dtype=torch.float).to(self.device)
            # 返回state下每个动作的q值
            action = self.q_net.forward(state).argmax().item()
        return action
        
        
        # 参数更新   
    def update(self,transition_dict):
        # 将state转换为一个形状为(1, 4)的二维张量，以便将其输入到网络中
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        # 将actions转换为二维张量
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        # 当前q值
        q_values = self.q_net.forward(states).gather(1, actions)
        # 下个状态的最大q值
        max_next_q_values = self.taget_q_net.forward(next_states).max(1)[0].view(-1,1)
        # TD误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)
        # 均方误差损失函数
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()  
         # 反向传播更新参数
        dqn_loss.backward() 
        self.optimizer.step()

        if self.count % self.taget_update == 0:
            # 更新target网络
            self.taget_q_net.load_state_dict(self.q_net.state_dict()) 
        self.count += 1
