import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
np.bool8 = np.bool

      #经验回放池
class ReplayBuffer:
    def __init__(self,capacity):
        # collections.deque双端队列，支持从两端快速地添加和删除元素,当队列达到maxlen时移除最早的元素
        self.buffer = collections.deque(maxlen=capacity)
        
        # 将数据加入buffer
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

        #从buffer中采样数据，数据量为batch_size
    def sample(self,batch_size): 
        # 随机采样
        transitions = random.sample(self.buffer,batch_size)
        # 解包transition，将同一维度的元素聚合在一起,如所有state放在一个state列表中
        state,action,reward,next_state,done = zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done
    
        # 检查当前buffer中的数据量
    def size(self):
        return len(self.buffer)
    
      #一层隐藏层的神经网络
class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        # 调用torch.nn.Module父类的构造函数
        super(Qnet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,action_dim)
        
        # 隐藏层使用ReLU激活函数（去负为0取最大）
    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
