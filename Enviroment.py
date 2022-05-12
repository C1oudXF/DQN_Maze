import numpy as np

class node:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.reword = 0
        self.idDone = False

class maze:
    def __init__(self,data,image):
        self.mazeImage = image
        self.mazeData = data
        self.weight,self.hight =data.shape
        self.action_space = ['u','d','l','r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title = 'maze'
        self.currentNode = [-1,-1]
        self.isarrive = False
        self.isblock = False
        self.endNode = [-1,-1]
        self.reward = 0



    def getstartposition(self):
        self.isblock = False
        self.isarrive = False
        for i in range(self.weight):
            for j in range(self.hight):
                if self.mazeData[i][j] == 2:
                    self.currentNode = [i,j]
                    return self.currentNode
        return None

    def getendPosition(self):
        for i in range(self.weight):
            for j in range(self.hight):
                if self.mazeData[i][j] == 3:
                    self.endNode = [i,j]
                    return self.endNode
        return None


    def step(self,action):
        #print(self.weight,self.hight)
        # self.mazeImage[self.currentNode[0]][self.currentNode[1]][0] = 255
        # self.mazeImage[self.currentNode[0]][self.currentNode[1]][1] = 255
        # self.mazeImage[self.currentNode[0]][self.currentNode[1]][2] = 255
        if action == 0:
            self.currentNode[1] -= 1
        elif action == 1:
            self.currentNode[1] += 1
        elif action == 2:
            self.currentNode[0] -= 1
        elif action == 3:
            self.currentNode[0] += 1
        # self.mazeImage[self.currentNode[0]][self.currentNode[1]][0] = 223
        # self.mazeImage[self.currentNode[0]][self.currentNode[1]][1] = 27
        # self.mazeImage[self.currentNode[0]][self.currentNode[1]][2] = 20
        # if self.mazeData[self.currentNode[0],self.currentNode[1]] == 0 and self.mazeData[self.currentNode[0],self.currentNode[1]] == 2:

        if self.currentNode[0] < 0 or \
               self.currentNode[0] >= self.weight or \
                self.currentNode[1] < 0 or \
                self.currentNode[1] >= self.hight:
            reward = -10
            isdone = True
            #print('离开地图')
        elif   self.mazeData[self.currentNode[0],self.currentNode[1]] == 1:
            reward = -10
            isdone = True
            #print('碰到障碍物')
        elif self.mazeData[self.currentNode[0],self.currentNode[1]] == 3:
            reward = 100
            isdone = True
            self.isarrive = True
            #print('到达终点')
        else:
            reward = 0
            isdone = False
        return [self.currentNode[0],self.currentNode[1]],reward,isdone
        #return self.mazeImage,reward,isdone



    #像素image的输出状态
    def pixstep(self,action):
        #print(self.weight,self.hight)
        self.mazeImage[self.currentNode[0]][self.currentNode[1]][0] = 255
        self.mazeImage[self.currentNode[0]][self.currentNode[1]][1] = 255
        self.mazeImage[self.currentNode[0]][self.currentNode[1]][2] = 255
        if action == 0:
            self.currentNode[1] -= 1
        elif action == 1:
            self.currentNode[1] += 1
        elif action == 2:
            self.currentNode[0] -= 1
        elif action == 3:
            self.currentNode[0] += 1

        if self.currentNode[0] < 0 or \
               self.currentNode[0] >= self.weight or \
                self.currentNode[1] < 0 or \
                self.currentNode[1] >= self.hight:
            self.reward = -10
            isdone = True
            #print('离开地图')
        else:
            self.mazeImage[self.currentNode[0]][self.currentNode[1]][0] = 20
            self.mazeImage[self.currentNode[0]][self.currentNode[1]][1] = 27
            self.mazeImage[self.currentNode[0]][self.currentNode[1]][2] = 223
            if self.mazeData[self.currentNode[0], self.currentNode[1]] == 1:
                self.reward = -10
                isdone = True
                self.isblock = True
                #print('碰到障碍物')
            elif self.mazeData[self.currentNode[0], self.currentNode[1]] == 3:
                self.reward = 100
                isdone = True
                self.isarrive = True
                # print('到达终点')
            else:
                isdone = False
        return self.mazeImage, self.reward, isdone
