import collections
import random
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import collections
from Envrioment import Maze


class RepalyBuffer():
    def __init__(self):
        self.buffer_limit = 50000
        self.buffer = collections.deque(maxlen= self.buffer_limit)
        self.buffer_counter = 0

    def put(self,transition):
        self.buffer.append(transition)
        """
        index = self.buffer_counter % self.buffer_limit
        self.buffer[index,:] = transition
        self.buffer_counter += 1
        """
    def sample(self,n):
        mini_batch = random.sample(self.buffer,n)
        s_list,a_list,r_list,s_next_list = [],[],[],[]

        for transition in mini_batch:
            s,a,r,s_ = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_next_list.append(s_)
        return torch.Tensor(s_list),torch.Tensor(a_list),\
               torch.Tensor(r_list),torch.Tensor(s_next_list)

    def size(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Actor, self).__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(in_features=input_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,out_features=output_size),
        )
    def forward(self,state):
        x= self.actor_net(state)
        x = torch.tanh(x)
        return x

class Critic(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Critic, self).__init__()
        self.critic_net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )

    def forward(self,state,action):
        inputs = torch.cat([state,action],1)
        x = self.critic_net(inputs)
        return x

class DDPG():
    def __init__(self,state_size,action_size,hidden_size=256,actor_lr = 0.001,critic_lr = 0.001,batch_size = 32):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor = Actor(self.state_size,self.hidden_size,self.action_size)
        self.actor_target = Actor(self.state_size,self.hidden_size,self.action_size)

        self.critic = Critic(self.state_size + self.action_size,self.hidden_size,self.action_size)
        self.critic_target = Critic(self.state_size + self.action_size,self.hidden_size,self.action_size)

        self.actor_optim = optim.Adam(self.actor.parameters(),lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(),lr=self.critic_lr)
        self.buffer = []

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.gamma = 0.99
        self.batch_size = batch_size
        self.memory = RepalyBuffer()

        self.memory2 = []
        self.learn_step_counter = 0
        self.replace_target_iter = 200
        self.cost_his_actor = []
        self.cost_his_critic = []


    def choose_action(self,state):
        state = torch.Tensor(state).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        return action

    def critic_learn(self,s0,a0,r1,s1):
        # 从actor_target通过状态获取对应的动作  detach()将tensor从计算图上剥离
        a1 = self.actor_target(s0).detach()
        # 删减一个维度  [b,1,1]变成[b,1]
        a0 = a0.squeeze(2)
        y_pred = self.critic(s0, a0)
        y_target = r1 + self.gamma * self.critic_target(s1, a1).detach()
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_target)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        self.cost_his_critic.append(loss.item())

 # actor网络的学习
    def actor_learn(self, s0, a0, r1, s1):
        loss = -torch.mean(self.critic(s0, self.actor(s0)))
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        self.cost_his_actor.append(loss.item())

    # 模型的训练
    def train(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
        # 随机采样出 batch_size 个(𝑠, 𝑎, 𝑟, 𝑠′)数据
        s0, a0, r, s_prime = self.memory.sample(self.batch_size)
        self.critic_learn(s0, a0, r, s_prime)
        self.actor_learn(s0, a0, r, s_prime)

        self.soft_update(self.critic_target, self.critic, 0.02)
        self.soft_update(self.actor_target, self.actor, 0.02)

    # target网络的更新
    def soft_update(self, net_target, net, tau):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def plot_cost(self):
        import matplotlib.pyplot as plt

        plt.plot(np.arange(len(self.cost_his_critic)), self.cost_his_critic)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



def main():

    env = Maze()
    agent = DDPG(state_size=4,
                 action_size=4,
                 hidden_size=128,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 batch_size=32
                 )

    for epispde in range(10000):
        env.reset()
        state = env.get_start_node()
        end = env.get_end_node()
        for step in range(100):
            state =
            action0 = agent.choose_action(state)




if __name__ == "__main__":
    main()




















