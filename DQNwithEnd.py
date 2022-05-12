import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import dataloader


#   #   #   #   #   #   #   #   #   #   #   #   #    #
#根据当前state与会获得不好奖励的state的相似度，来纠正reward。 #
#   #   #   #   #   #   #   #   #   #   #   #   #   #

class DQN_with_pred():
    def __init__(self,
                 dim_state,
                 n_actions,
                 batch_size=128,
                 batch_size2=32,
                 learning_rate=0.01,
                 epsilon=0.9,
                 gamma=0.9,
                 target_replace_iter=100,
                 memory_size=2000,
                 data_size = 2000):
        # 调用类内自写函数生成网络
        self.eval_net, self.target_net = self.bulid_Net(dim_state, n_actions), self.bulid_Net(dim_state, n_actions)
        self.predict_net = self.build_predict_network()

        self.dim_state = dim_state  # 状态维度
        self.n_actions = n_actions  # 可选动作数
        self.batch_size = batch_size        # 小批量梯度下降，每个“批”的size
        self.batch_size2 = batch_size2      # 小批量梯度下降，每个“批”的size
        self.learning_rate = learning_rate  # 学习率
        self.epsilon = epsilon  # 贪婪系数
        self.gamma = gamma      # 回报衰减率
        self.memory_size = memory_size                  # 记忆库的规格
        self.data_size = data_size                      # 启发式网络训练库的规格
        self.data_count = 0                             #启发式网络的训练索引
        self.taget_replace_iter = target_replace_iter   # target网络延迟更新的间隔步数
        self.learn_step_counter = 0     # 在计算隔n步跟新的的时候用到
        self.memory_counter = 0         # 用来计算存储索引
        #self.memory = np.zeros((self.memory_size, self.dim_state * 2 + 2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)    # 网络优化器
        self.loss_func = nn.MSELoss()   # 预测网络的损失函数
        self.feel_loss_func = nn.MSELoss()              #感觉网络的损失函数
        self.feel_optimizer = torch.optim.Adam(self.predict_net.parameters(), lr=self.learning_rate)    # 感觉网络的优化器
        self.build_memory_pool(self.memory_size)
        self.bulid_predict_pool(data_size=data_size)
        self.reward_list = [-10,10]

    def build_memory_pool(self,memory_size):
        self.state_memory = np.zeros((memory_size,3,30,30))
        self.a_r_memory = np.zeros((memory_size,2))
        self.next_state_memory = np.zeros((memory_size,3,30,30))



    # 选择动作
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:  # greedy概率有eval网络生成动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1]
            action = int(action)
        else:  # （1-greedy）概率随机选择动作
            action = np.random.randint(0, self.n_actions)
        return action

    # 测试时选择动作
    def choose_action_test(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1]
        action = int(action)
        return action

    # 学习，更新网络参数
    def learn(self):
        # 目标网络参数更新（经过self.taget_replace_iter步之后，为target_net网络更新参数）
        if self.learn_step_counter % self.taget_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 从记忆库中提取一个batch的数据
        data_size = self.memory_size if self.memory_counter>self.memory_size else self.memory_counter

        sample_index = np.random.choice(data_size, self.batch_size)
        #b_memory = self.memory[sample_index, :]
        b_s_memory = self.state_memory[sample_index,:]
        b_s = torch.FloatTensor(b_s_memory)
        #b_s = torch.FloatTensor(b_memory[:, :self.dim_state])
        b_ar_memory = self.a_r_memory[sample_index,:]
        b_a = torch.LongTensor(b_ar_memory[:,:1].astype(int))
        b_r = torch.FloatTensor(b_ar_memory[:, 1:2])
        #b_s_ = torch.FloatTensor(b_memory[:, -self.dim_state:])
        b_nexts_memory = self.next_state_memory[sample_index,:]
        b_s_ = torch.FloatTensor(b_nexts_memory[:,:self.dim_state])


        """
        # 将b_s和b_s_还原成image的状态（3*30*30）
        print('b_s:',b_s)
        a,b,c=0,0,0
        b_s1 = []
        count = 0
        bs = np.zeros((3,30,30))
        print(len(bs))
        for _ in b_s:
            a = 0
            b = 0
            c = 0
            for i in _:
                bs[a][b][c] = i
                c += 1
                if c == 30 :
                    b += 1
                    c = 0
                if b == 30 :
                    a += 1
                    b = 0
            b_s1.append(bs)
        b_s1 = torch.FloatTensor(b_s1)
        print('b_s1:',b_s1)
        """
        # 获得q_eval、q_target，计算loss
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        # 反向传递，更新eval网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 存储一步的信息到记忆库
    def store_transition(self, s, a, r, s_):
        #transition = np.hstack((s, [a, r], s_))
        # 存储记忆（如果第一轮存满了，就覆盖存入）
        index = self.memory_counter % self.memory_size
        transition_s = s
        transition_a_r = np.hstack(([a,r]))
        transition_s_ = s_
        self.state_memory[index,:] = transition_s
        self.a_r_memory[index,:] = transition_a_r
        self.next_state_memory[index,:] = transition_s_
        #self.memory[index, :] = transition
        self.memory_counter += 1

    # 构建网络
    def bulid_Net(self, dim_state, n_actions):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, n_actions),
            torch.nn.Softmax(dim=1)
        )

#--------------------------------------------------------------------#

    def bulid_predict_pool(self,data_size):
        self.end_state_memory_pool = np.zeros((data_size,3,30,30))#[索引,3,30,30]
        self.end_state_r_memory_pool = np.zeros((data_size,1))#[索引,reward]

    def store_end_transition(self,state,r):
        index = self.data_count % self.data_size
        transition_end_s = state
        transition_end_r = r
        self.end_state_memory_pool[index,:] = transition_end_s
        self.end_state_r_memory_pool[index,:] = transition_end_r
        self.data_count += 1

    def build_predict_network(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 2),
            torch.nn.Softmax(dim=1)
        )

    def learn_predict_net(self):
        #经验池中共有多少数据
        data_size = self.data_size if self.data_count > self.data_size else self.data_count
        sample_index = np.random.choice(data_size, self.batch_size2)
        #print(sample_index)
        for index in sample_index:
            p_s_memory = self.end_state_memory_pool[index,:]
            p_s_memory = torch.FloatTensor(p_s_memory)
            p_s_memory = torch.unsqueeze(p_s_memory,0)
            p_r = self.end_state_r_memory_pool[index, :]
            p_r = torch.FloatTensor(p_r)
            #print('learn_state:',p_s_memory)
            output = self.predict_net.forward(p_s_memory)
            loss = self.feel_loss_func(output,p_r)
            self.feel_optimizer.zero_grad()
            loss.backward()
            self.feel_optimizer.step()


    def correct_reward(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        #print('input_state:',state)
        similer_value = self.predict_net.forward(state)
        similer_value = similer_value.tolist()
        #print('similer_value:',similer_value)
        #print(self.reward_list)
        reward_correct = similer_value[0][0] * self.reward_list[0] + similer_value[0][1] * self.reward_list[1]
        return reward_correct



