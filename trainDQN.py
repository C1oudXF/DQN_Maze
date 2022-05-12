import os
import cv2
import numpy as np
import random

import torch
from DQNwithEnd import DQN_with_pred
from Enviroment import maze
#from Agent import agent
import shutil
from PytorchDQN import DQN
from PIL import Image


arriveCount = 0
blockcounte = 0
train = 5
rewardnumber = 0
STEPS = 1000
EPISODES = 100000
imageSize = [30,30,3]
limit = 30*30
#images = np.zeros([imageSize[0],imageSize[1],imageSize[2]])
path = "trainImages/"
files = os.listdir(path)
#将file中的图像转换为强化学习环境
flag = np.zeros([30,30])
rewardlist = []
input2image = np.zeros((30,30,3))
input2image4 = np.zeros((3,30,30))
output2image4 = np.zeros((3,30,30))
stepnumber = 0
sumreward = 0
episode_reward = 0
episode_reward_list = []
max_reward = -9999
folderName = 'DQNTrainImage'
# if os.path.exists(folderName):
# 	shutil.rmtree(folderName)
# os.mkdir(folderName)

#image = cv2.imread(path + "/" + file)
# 从训练图片中随机抽取一张
# sampleImage = random.randint(0,index)
"""
只用一个地图训练
"""
# sampleImage = 6
# file = files[sampleImage]
# image = cv2.imread(path + "/" + file)
# # 有image到map
# for i in range(25):
#     for j in range(25):
#         if image[i][j][0] == 255:
#             # 路
#             flag[i][j] = 0
#         elif image[i][j][0] == 0:
#             # 障碍物
#             flag[i][j] = 1
#         elif image[i][j][0] == 223:
#             # 起点
#             flag[i][j] = 2
#         elif image[i][j][0] == 40:
#             # 终点
#             flag[i][j] = 3
# # 转换为环境类
# env = maze(flag)
# state = env.getstartposition()
# print(state)

"""
结束
"""
#agent = agent(env.n_actions,env.n_features)
#agent = agent()
#print(flag)
#index = len(files)
# print(index)
    #每个episode随机抽取一个地图来训练
"""
if  train == 1 :
    agent = agent()
    index = len(files)
    #train 1 : input：位置编码
    for episode in range(EPISODES):

        #随机抽地图训练
        
        #从训练图片中随机抽取一张
        #sampleImage = random.randint(0,index)
        sampleImage = 1
        file = files[sampleImage]
        image = cv2.imread(path + "/" + file)
        #有image到map
        for i in range(30):
            for j in range(30):
                if image[i][j][0] == 255:
                    # 路
                    flag[i][j] = 0
                elif image[i][j][0] == 0:
                    # 障碍物
                    flag[i][j] = 1
                elif image[i][j][0] == 223:
                    # 起点
                    flag[i][j] = 2
                elif image[i][j][0] == 40:
                    # 终点
                    flag[i][j] = 3
        #转换为环境类
        #print('flag:',flag)
        #print('image',image)
        env = maze(flag,image)
        state = env.getstartposition()
        step = 0
        while True:
            state = np.array(state)
            #print(state)
            action = agent.choose_action(state)
            next_state,reward,isDone = env.step(action)
            if env.isblock == 1 :
                blockcounte += 1
            rewardlist.append(reward)
            agent.store_transition(state,reward,action,next_state)
            #print('agentg memory count: ',agent.memory_counter)
            #print('agentg learn step count: ', agent.learn_step_counter)
            if agent.memory_counter > 200 and stepnumber % 5 == 0:
                agent.learn()
                #print('agent.learn\n')
            state = next_state
            step += 1
            stepnumber += 1
            if isDone:
                break
            #print(state,action)

        if len(rewardlist) >= 200:
            for _ in rewardlist:
                sumreward += _
                rewardnumber += 1
            mainreward = sumreward / rewardnumber
            print('stepnumber:',stepnumber)
            print('mianreward:',mainreward)
            print('-----------------------------------')
            print('\n')
            sumreward = 0
            rewardnumber = 0
            rewardlist.clear()




#train2 ： input 图片像素
if train == 2 :
    agent = agent()
    index = len(files)
    for episode in range(EPISODES):
        #从训练图片中随机抽取一张
        #sampleImage = random.randint(0,index-1)
        sampleImage = 6
        file = files[sampleImage]
        image = cv2.imread(path + "/" + file)
        #print('image:',image)
        #有image到map
        for i in range(30):
            for j in range(30):
                if image[i][j][0] == 255:
                    # 路
                    flag[i][j] = 0
                elif image[i][j][0] == 0:
                    # 障碍物
                    flag[i][j] = 1
                elif image[i][j][0] == 223:
                    # 起点
                    flag[i][j] = 2
                elif image[i][j][0] == 40:
                    # 终点
                    flag[i][j] = 3
        #转换为环境类
        env = maze(flag,image)
        pixstate = image.flatten()
        start_state = env.getstartposition()
        #print(len(pixstate))
        step = 0
        while True:
            pixstate = np.array(pixstate)
            #print(pixstate)
            for x in range(30):
                for y in range(30):
                    for z in range(3):
                        input2image[x][y][z] = pixstate[z + y * 3 + x * 3 *30 ]

            #print('input2image:',input2image)
            cv2.imwrite(os.path.join(folderName, "image_" + str(episode) + "_" + str(step) + ".png"),input2image)
            action = agent.choose_action(pixstate)
            #print(action)
            next_state,reward,isDone = env.pixstep(action)
            if env.isblock == True :
                blockcounte += 1
            if env.isarrive== True:
                arriveCount += 1
            rewardlist.append(reward)
            agent.store_transition(pixstate,reward,action,next_state)
            #print('agentg memory count: ',agent.memory_counter)
            #print('agentg learn step count: ', agent.learn_step_counter)
            if agent.memory_counter > 200 and stepnumber % 5 == 0:
                agent.learn()
                #print('agent.learn\n')
            pixstate = next_state
            step += 1
            stepnumber += 1
            if isDone:
                break

            #print(state,action)
        #print(step)
        if len(rewardlist) >= 200:
            for _ in rewardlist:
                sumreward += _
                rewardnumber += 1
            mainreward = sumreward / rewardnumber
            print('stepnumber:',stepnumber)
            print('mianreward:',mainreward)
            print('episode:',episode)
            print('step',step)
            print('-----------------------------------')
            print('\n')
            sumreward = 0
            blockcounte = 0
            rewardnumber = 0
            rewardlist.clear()




if  train == 3 :
    agent = agent()
    index = len(files)
    #train 1 : input：位置编码
    for episode in range(EPISODES):
        #从训练图片中随机抽取一张
        #sampleImage = random.randint(0,index)
        sampleImage = 1
        file = files[sampleImage]
        image = cv2.imread(path + "/" + file)
        #有image到map
        for i in range(30):
            for j in range(30):
                if image[i][j][0] == 255:
                    # 路
                    flag[i][j] = 0
                elif image[i][j][0] == 0:
                    # 障碍物
                    flag[i][j] = 1
                elif image[i][j][0] == 223:
                    # 起点
                    flag[i][j] = 2
                elif image[i][j][0] == 40:
                    # 终点
                    flag[i][j] = 3
        #转换为环境类
        #print('flag:',flag)
        #print('image',image)
        env = maze(flag,image)
        state = env.getstartposition()
        #print(state)
        endstate = env.getendPosition()
        #print(endstate)
        step = 0
        while True:
            train3input = state
            #train3input.extend(endstate)
            #print(train3input)
            train3input = np.array(train3input)
            train3input = np.append(train3input,endstate)
            state = np.array(state)
            #print(state)

            #print(train3input)

            action = agent.choose_action(train3input)
            next_state,reward,isDone = env.step(action)
            #print('next_state0:',next_state)
            if env.isblock == 1 :
                blockcounte += 1
            rewardlist.append(reward)
            nextinput = next_state
            nextinput.extend(endstate)
            agent.store_transition(train3input,reward,action,nextinput)
            #print('agentg memory count: ',agent.memory_counter)
            #print('agentg learn step count: ', agent.learn_step_counter)
            if agent.memory_counter > 200 and stepnumber % 5 == 0:
                agent.learn()
                #print('agent.learn\n')
            next_state = np.delete(next_state, -1)
            next_state = np.delete(next_state, -1)
            state = next_state
            #print('next state:',next_state)
            step += 1
            stepnumber += 1
            if isDone:
                break
            #print(state,action)

        if len(rewardlist) >= 200:
            for _ in rewardlist:
                sumreward += _
                rewardnumber += 1
            mainreward = sumreward / rewardnumber
            print('stepnumber:',stepnumber)
            print('mianreward:',mainreward)
            print('-----------------------------------')
            print('\n')
            sumreward = 0
            rewardnumber = 0
            rewardlist.clear()
"""


#train4 ： input 图片像素 + CNN
if train == 4 :

    agent = DQN(n_actions=4,dim_state=2700)
    for episode in range(EPISODES):
        """
        随机抽地图训练
        """
        #从训练图片中随机抽取一张
        #sampleImage = random.randint(0,index-1)
        #sampleImage = 6
        #file = files[sampleImage]
        file = 'trainImages/image_1_0.png'
        image = np.array(Image.open(file).convert("RGB"))
        #image = cv2.imread(path + "/" + file)
        #print('image:',image)
        #有image到map
        for i in range(30):
            for j in range(30):
                if image[i][j][0] == 255:
                    # 路
                    flag[i][j] = 0
                elif image[i][j][0] == 0:
                    # 障碍物
                    flag[i][j] = 1
                elif image[i][j][2] == 223:
                    # 起点
                    flag[i][j] = 2
                elif image[i][j][2] == 40:
                    # 终点
                    flag[i][j] = 3
        #转换为环境类
        env = maze(flag,image)
        #print(flag)
        pixstate = image.flatten()
        start_state = env.getstartposition()
        #print(len(pixstate))
        step = 0
        for x in range(30):
            for y in range(30):
                for z in range(3):
                    input2image4[z][x][y] = image[x][y][z]
                    # input2image4[z][x][y] = pixstate[z + y * 3 + x * 3 *30 ]
        while True:
            #print(pixstate)
            #print('input2image:',input2image)
            for x in range(30):
                for y in range(30):
                    for z in range(3):
                        image[x][y][z] = input2image4[z][x][y]
            #cv2.imwrite(os.path.join(folderName, "image_" + str(episode) + "_" + str(step) + ".png"),image)
            action = agent.choose_action(input2image4)
            #print(action)
            next_state,reward,isDone = env.pixstep(action)
            #print(isDone)
            #print('第一个：',next_state)
            if env.isblock == True :
                blockcounte += 1
            if env.isarrive== True :
                arriveCount += 1
            rewardlist.append(reward)
            for x in range(30):
                for y in range(30):
                    for z in range(3):
                        output2image4[z][x][y] = next_state[x][y][z]
            #print('第二个:',output2image4)
            agent.store_transition(input2image4,action,reward,output2image4)
            #print('agentg memory count: ',agent.memory_counter)
            #print('agentg learn step count: ', agent.learn_step_counter)
            if agent.memory_counter > 200 and episode % 5 == 0:
                agent.learn()
                #print('agent.learn\n')
            input2image4 = output2image4
            step += 1
            stepnumber += 1
            if isDone:
                for _ in rewardlist:
                    episode_reward += _
                if episode_reward > max_reward :
                    max_reward = episode_reward
                rewardlist.clear()
                episode_reward_list.append(episode_reward)
                episode_reward = 0
                break

            #print(state,action)
        #print(step)
        if episode % 200 == 0 and episode != 0 :
            for _ in episode_reward_list:
                sumreward += _
            mainreward = sumreward / 200
            print('episode:',episode)
            print('这200个episode的总奖励：',sumreward)
            print('这200个episode的平均奖励：',mainreward)
            print('目前所获得的最高奖励：',max_reward)
            #print('step',step)
            print('-----------------------------------')
            sumreward = 0
            blockcounte = 0
            episode_reward_list.clear()
    torch.save(agent.eval_net,"troch_3cnn_evalnet_2022_5_12.pth")
    torch.save(agent.target_net,"torch_3cnn_targetnet_2022_5_12.pth")


#train5 ： input 图片像素 + CNN + 对当前状态是否接近于失败的感觉。
if train == 5 :

    agent = DQN_with_pred(n_actions=4,dim_state=2700)
    for episode in range(EPISODES):
        """
        随机抽地图训练
        """
        #从训练图片中随机抽取一张
        #sampleImage = random.randint(0,index-1)
        #sampleImage = 6
        #file = files[sampleImage]
        file = 'trainImages/image_1_0.png'
        image = np.array(Image.open(file).convert("RGB"))
        #image = cv2.imread(path + "/" + file)
        #print('image:',image)
        #有image到map
        for i in range(30):
            for j in range(30):
                if image[i][j][0] == 255:
                    # 路
                    flag[i][j] = 0
                elif image[i][j][0] == 0:
                    # 障碍物
                    flag[i][j] = 1
                elif image[i][j][2] == 223:
                    # 起点
                    flag[i][j] = 2
                elif image[i][j][2] == 40:
                    # 终点
                    flag[i][j] = 3
        #转换为环境类
        env = maze(flag,image)
        #print(flag)
        pixstate = image.flatten()
        start_state = env.getstartposition()
        #print(len(pixstate))
        step = 0
        for x in range(30):
            for y in range(30):
                for z in range(3):
                    input2image4[z][x][y] = image[x][y][z]
                    # input2image4[z][x][y] = pixstate[z + y * 3 + x * 3 *30 ]
        while True:
            #print(pixstate)
            #print('input2image:',input2image)
            for x in range(30):
                for y in range(30):
                    for z in range(3):
                        image[x][y][z] = input2image4[z][x][y]
            #cv2.imwrite(os.path.join(folderName, "image_" + str(episode) + "_" + str(step) + ".png"),image)
            action = agent.choose_action(input2image4)
            #print(action)
            next_state,reward,isDone = env.pixstep(action)
            for x in range(30):
                for y in range(30):
                    for z in range(3):
                        output2image4[z][x][y] = next_state[x][y][z]
            #对next_state做一下判断，看他是否会导致角色死亡
            reward_correct = agent.correct_reward(output2image4)
            original_reward = reward
            reward = reward + reward_correct
            #print(isDone)
            #print('第一个：',next_state)
            if env.isblock == True :
                blockcounte += 1
            if env.isarrive== True :
                arriveCount += 1
            rewardlist.append(reward)
            #print('第二个:',output2image4)
            agent.store_transition(input2image4,action,reward,output2image4)
            #print('agentg memory count: ',agent.memory_counter)
            #print('agentg learn step count: ', agent.learn_step_counter)
            if agent.memory_counter > 200 and step % 5 == 0:
                agent.learn()
                #print('agent.learn\n')
            input2image4 = output2image4
            step += 1
            stepnumber += 1
            if isDone:
                agent.store_end_transition(state=input2image4,r=original_reward)
                for _ in rewardlist:
                    episode_reward += _
                if episode_reward > max_reward :
                    max_reward = episode_reward
                rewardlist.clear()
                episode_reward_list.append(episode_reward)
                episode_reward = 0
                break

            #print(state,action)
        #print(step)
        if agent.data_count > 200 and episode % 5 == 0:
            agent.learn_predict_net()

        if episode % 200 == 0 and episode != 0 :
            for _ in episode_reward_list:
                sumreward += _
            mainreward = sumreward / 200
            print('episode:',episode)
            print('这200个episode的总奖励：',sumreward)
            print('这200个episode的平均奖励：',mainreward)
            print('目前所获得的最高奖励：',max_reward)
            #print('step',step)
            print('-----------------------------------')
            sumreward = 0
            blockcounte = 0
            episode_reward_list.clear()
    torch.save(agent.eval_net,"/train5_model/troch_3cnn_evalnet_2022_5_12.pth")
    torch.save(agent.target_net,"/train5_model/torch_3cnn_targetnet_2022_5_12.pth")






#saver = tf.compat.v1.train.Saver()
#saver.restore(agent.sess,'/DQNLearning/DQN_Modle2.ckpt')
#save_path = saver.save(agent.sess,'/DQNLearning/DQN_Modle.ckpt')
print('game over')
print('共到达终点：',arriveCount)