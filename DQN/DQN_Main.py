import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import DQN_Net
import DQN_Algorithm
import rl_utils
np.bool8 = np.bool

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 初始化环境，定义环境实例
env_name = 'CartPole-v0'
env = gym.make(env_name)

random.seed(0)
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)

replay_buffer = DQN_Net.ReplayBuffer(buffer_size)

# 获取环境状态空间的维度
state_dim = env.observation_space.shape[0]
# 获取动作空间的维度
action_dim = env.action_space.n

agent = DQN_Algorithm.DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                # 找到初始状态
                state = env.reset()
                # 由于env.reset()返回值是一个元组，其中第一个元素是包含状态的NumPy数组，第二个元素是额外的信息字典，我们需要取第一个Numpy数组
                state = state[0]
                done = False
                while not done:
                    # 在state状态根据ε-Greedy选择一个动作
                    action = agent.take_action(state)
                    next_state,reward,done,truncated,_ = env.step(action)
                    done = done or truncated
                    replay_buffer.add(state,action,reward,next_state,done)
                    state = next_state
                    episode_return += reward
                    # 当replay_buffer中的数据超过设定的值后，才开始训练
                    if replay_buffer.size() > minimal_size:
                        s,a,r,ns,d = replay_buffer.sample(batch_size)
                        #将采样的值加入transition_dict中
                        transition_dict = {
                                            'states' : s,
                                            'actions' : a,
                                            'rewards' : r,
                                            'next_states' : ns,
                                            'dones' : d
                                            }
                        agent.update(transition_dict)
                # 在一个episode完成后在return_list中添加这一段的return
                return_list.append(episode_return)
                
                # 每10个episode打印一次统计信息
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                # 每完成一个episode，进度条就会更新一步
                pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()



