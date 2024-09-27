import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import D_DQN_Net
np.bool8 = np.bool

      # DQN算法，包括Double DQN和Dueling DQN
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device,dqn_type = 'VanillaDQN'):
        self.action_dim = action_dim

        # Dueling DQN采取不同的网络框架
        if dqn_type == 'DuelingDQN':
           self.q_net = D_DQN_Net.VAnet(state_dim,hidden_dim,self.action_dim).to(device)
           self.target_q_net = D_DQN_Net.VAnet(state_dim,hidden_dim,self.action_dim).to(device)
        # 另一套采取DQN网络框架
        else:
            self.q_net = D_DQN_Net.Qnet(state_dim,hidden_dim,self.action_dim).to(device)
            self.q_net = D_DQN_Net.Qnet(state_dim,hidden_dim,self.action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self,state):
           # 生成一个[0,action_dim-1]的随机整数,若小于ε，则随机选取一个action
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state],dtype=torch.float).to(self.device)
            # 返回使得q值最大的动作
            # item()将张量中的单个元素转为Python标量
            action = self.q_net.forward(state).argmax().item()
        return action
    
        # 寻找最大的q值
    def max_q_value(self,state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()
    
    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        # actions转换为张量后仍然是一维，需要通过view(-1,1)reshape一下成为二维
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
        # 在动作维度，根据采取的动作的标号选取每个采样state的q
        q_values = self.q_net(states).gather(1,actions)
        
        # 判断使用的是DoubleDQN还是普通DQN
        # DoubleDQN先选取能取到最大q的action，然后用action更新目标网络的q
        # 普通DQN是直接获取最大的q更新目标网络
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net.forward(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net.forward(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net.forward(next_states).max(1)[0].view(-1, 1)

        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)
          
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1