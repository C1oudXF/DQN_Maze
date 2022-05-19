import numpy as np
from PIL import Image
import cv2
import os
folderName = 'output_image'

file = 'image_1_1.png'

class Maze:
    def __init__(self,image):
        self.Maze_image = image
        self.Maze_flag = np.zeros((30,30))
        self.start_node = np.zeros((2))
        self.current_node = np.zeros((2))
        self.end_node = np.zeros((2))
        self.reset()



    def reset(self):
        while True:
            end_x = np.random.randint(1, 29)
            end_y = np.random.randint(1, 29)
            if end_x != 5 and end_y != 3 :
                break
        self.Maze_image = np.array(Image.open(file).convert("RGB"))
        self.Maze_image[end_x][end_y][0] = 20
        self.Maze_image[end_x][end_y][1] = 222
        self.Maze_image[end_x][end_y][2] = 40
        for i in range(30):
            for j in range(30):
                if self.Maze_image[i][j][0] == 255:
                    # 路
                    self.Maze_flag[i][j] = 0
                elif self.Maze_image[i][j][0] == 0:
                    # 障碍物
                    self.Maze_flag[i][j] = 1
                elif self.Maze_image[i][j][2] == 223:
                    # 起点
                    self.Maze_flag[i][j] = 2
                    self.start_node = [i,j]
                    self.current_node = [i,j]
                    #print(self.current_node)
                    # print('有起点')
                elif self.Maze_image[i][j][2] == 40:
                    # 终点
                    #print('有终点')
                    self.Maze_flag[i][j] = 3
                    self.end_node = [i,j]


    def show_map(self,episode,step):
        cv2.imwrite(os.path.join(folderName, "image_" + str(episode) + "_" + str(step) + ".png"), self.Maze_image)

    def get_start_node(self):
        return self.start_node

    def get_end_node(self):
        return self.end_node

    def get_current_node(self):
        return self.current_node

    def step(self,action):
        distance =self.current_node[0]-self.end_node[0] * self.current_node[0]-self.end_node[0] + \
                        self.current_node[1]-self.end_node[1] * self.current_node[1]-self.end_node[1]
        # self.Maze_image[self.current_node[0]][self.current_node[1]][0] = 0
        # self.Maze_image[self.current_node[0]][self.current_node[1]][1] = 0
        # self.Maze_image[self.current_node[0]][self.current_node[1]][2] = 0
        if action == 0 :
            self.current_node[1] -= 1
        elif action == 1:
            self.current_node[1] += 1
        elif action == 2:
            self.current_node[0] -= 1
        elif action == 3:
            self.current_node[0] += 1

        next_distance = self.current_node[0] - self.end_node[0] * self.current_node[0] - self.end_node[0] + \
                   self.current_node[1] - self.end_node[1] * self.current_node[1] - self.end_node[1]
        self.Maze_image[self.current_node[0]][self.current_node[1]][0] = 20
        self.Maze_image[self.current_node[0]][self.current_node[1]][1] = 27
        self.Maze_image[self.current_node[0]][self.current_node[1]][2] = 223

        if self.current_node == self.end_node:
            reward = 10
            isdone = True
        elif self.Maze_flag[self.current_node[0],self.current_node[1]] == 1:
            reward = 0
            isdone = True
        else:
            isdone = False
            reward = 0
        return self.current_node,reward,isdone


