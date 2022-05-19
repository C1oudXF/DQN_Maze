import torch
import numpy as np


class DlAgent:
    def __init__(self,
                 dim_state,
                 n_actions,
                 pix_dim,
                 batch_size=128,
                 learning_rate=0.01,
                 epsilon=0.8,
                 gamma=0.9,
                 target_replace_iter=100,
                 Advance_memory_size=2000,
                 path_memory_size = 2000,
                 ):
        #超参数初始化
        self.dim_state = dim_state
        self.n_actions = n_actions
        self.pix_dim = pix_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.Advance_memory_size = Advance_memory_size
        self.path_memory_size = path_memory_size
        self.path_memory_counter = 0
        self.path_learn_step_counter = 0


        # 全局路径引导
            # 构建网络模型
        self.path_value_net = self.build_path_net(self.dim_state,self.n_actions)
        self.path_target_net = self.build_path_net(self.dim_state,self.n_actions)
            #构建经验回访池
        self.build_path_memory_pool(self.path_memory_size)
            #构建全局路径网络的优化器
        self.pathnet_loss_func = torch.nn.MSELoss()
        self.pathnet_optimizer = torch.optim.Adam(self.path_value_net.parameters(),lr=self.learning_rate)

        #局部路径避障
            # 构建网络模型
        #self.Advance_value_net
        #self.Advance_target_net

        #

    def build_path_memory_pool(self,memory_size):
        self.path_memory_state = np.zeros((memory_size,self.dim_state))
        self.path_memory_action = np.zeros((memory_size, 1))
        self.path_memory_reward = np.zeros((memory_size, 1))
        self.path_memory_next_state = np.zeros((memory_size,self.dim_state))

    def store_path_memory(self,s,a,r,s_):
        index =self.path_memory_counter % self.path_memory_size
        transition_s = s
        transition_a = a
        transition_r = r
        transition_s_ = s_

        self.path_memory_state[index,:] = transition_s
        self.path_memory_action[index,:] = transition_a
        self.path_memory_reward[index,:] = transition_r
        self.path_memory_next_state[index,:] = transition_s_

        self.path_memory_counter += 1


    #引导到终点的路径规划模型    linear+linear+linear
    def build_path_net(self,state_dim,action_dim):
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=state_dim,out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128,out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128,out_features=action_dim),
            torch.nn.Softmax()
        )

    def choose_next_path(self,state):
        #print('state:',state)
        x = torch.unsqueeze(torch.FloatTensor(state),0)
        if np.random.uniform() < self.epsilon:
            actions_value = self.path_value_net.forward(x)
            action = torch.max(actions_value,1)[1]
            action = int(action)
        else:
            action = np.random.randint(0,self.n_actions)
        return action


    def learn_path_net(self):
        if self.path_learn_step_counter % self.target_replace_iter == 0 :
            self.path_target_net.load_state_dict(self.path_value_net.state_dict())
            self.path_learn_step_counter += 1

        if self.path_memory_counter > self.path_memory_size:
            data_size = self.path_memory_size
        else:
            data_size = self.path_memory_counter

        sample_index = np.random.choice(data_size,self.batch_size)
        b_s = torch.FloatTensor(self.path_memory_state[sample_index,:])
        b_a = torch.LongTensor(self.path_memory_action[sample_index,:].astype(int))
        b_r = torch.FloatTensor(self.path_memory_reward[sample_index,:])
        b_s_ = torch.FloatTensor(self.path_memory_next_state[sample_index,:])

        q_eval = self.path_value_net(b_s).gather(1,b_a)
        q_next = self.path_target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size,1)
        loss = self.pathnet_loss_func(q_eval,q_target)

        self.pathnet_optimizer.zero_grad()
        loss.backward()
        self.pathnet_optimizer.step()

