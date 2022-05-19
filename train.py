import os
import cv2
import numpy as np
import random
from PIL import Image
import torch
from DL import DlAgent
from Envrioment import Maze


EPISODE = 100000
sumreward = 0
#读取地图数据
file = 'image_1_0.png'
image = np.array(Image.open(file).convert("RGB"))

#生成agent
agent = DlAgent(dim_state=4,n_actions=4,pix_dim=3)
Maze = Maze(image)
end_position = Maze.get_end_node()
for episode in range(EPISODE):
    #print('episode:',episode)
    step = 0
    episode_reward = 0
    input = Maze.get_start_node()
    input = np.append(input,end_position)
    while True:
        action = agent.choose_next_path(input)
        next_state,reward,isdone = Maze.step(action)
        if episode % 500 == 0:
        #Maze.show_map(episode, step)
            Maze.show_map(episode,step)
            print('action',action)
        episode_reward += reward
        next_input = np.append(next_state,end_position)
        #print('next_input',next_input)
        agent.store_path_memory(input,action,reward,next_input)
        if agent.path_memory_counter > 500 and episode % 5 == 0 :
            agent.learn_path_net()
            #print('agent get learned')
        input = next_input

        if isdone == True:
            Maze.reset()
            sumreward += episode_reward
            break
        step += 1
    if episode % 200 == 0:
        print('episode:',episode)
        mainreward = sumreward / 200
        print('main reward in this 200 episode:',mainreward)
        sumreward = 0



torch.save(agent.path_value_net,'./model/path_value_model_2022_5_19.pth')
torch.save(agent.path_target_net,'./model/path_target_model_2022_5_19.pth')




