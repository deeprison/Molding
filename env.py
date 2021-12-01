import numpy as np
import cv2
from data.main import *
from copy import deepcopy

class env:
    def __init__(self,
                 image_list:list = None,
                 image_size:int = 20,
                 prob_cutout:int = 0,
                 num_images:int = 5,
                 start_position:tuple = None,
                 start_direction:tuple = None,
                 ):
        # images
        self.image_list = image_list
        if self.image_list == None:
            self.image_list = self.generate_images(image_size, prob_cutout, num_images)
                
        # info
        '''
        state_dim : 이미지 크기
        n_action : action 수
        actions : 진행 방향을 기준으로
        {0 : 동작 안함, 1 : 좌측 상단, 2 : 좌측, 3 : 좌측 하단,
        4 : 우측 상단, 5 : 우측, 6 : 우측 하단}
        '''
        self.state_dim = self.image_list[0].shape
        self.action_space = 7
        self.all_directions = [(0,-1),  #↑
                               (1,-1),  #↗
                               (1,0),   #→
                               (1,1),   #↘
                               (0,1),   #↓
                               (-1,1),  #↙
                               (-1,0),  #←
                               (-1,-1)] #↖
        
        # initialize
        self.current_image = deepcopy(self.image_list[np.random.randint(len(self.image_list))])
        self.image_size = self.current_image.shape[0]

        candidate_positions = [(0,0), (0,self.image_size-1),(self.image_size-1,0),(self.image_size-1, self.image_size-1)]
        
        self.start_position = start_position
        if self.start_position == None:
            self.start_position = candidate_positions[np.random.choice(len(candidate_positions), 1)[0]]
        self.current_position = list(self.start_position)

        self.start_direction = start_direction
        if self.start_direction == None:
            if self.current_position == list(candidate_positions[0]): 
                direction_index = np.random.choice([2, 4],1)
            elif self.current_position == list(candidate_positions[1]): 
                direction_index = np.random.choice([0, 2],1)
            elif self.current_position == list(candidate_positions[2]): 
                direction_index = np.random.choice([4, 6],1)
            elif self.current_position == list(candidate_positions[3]): 
                direction_index = np.random.choice([0, 6],1)
            self.start_direction = self.all_directions[direction_index[0]]

        self.current_direction = self.start_direction
        self.time = 0
        
        self.fill_up_done = -(self.image_size**2) + 2*np.sum(self.current_image)
        self.time_end_done = (self.image_size**2)*2
    
    def generate_images(self, img_size, prob_cutout, num_images):
        image_list = []
        for i in range(num_images):
            image_list.append(square_gen(img_size, prob_cutout))
        return image_list
    
    def reset(self):
        image_index = np.random.randint(len(self.image_list))
        self.current_image = self.image_list[image_index]
        self.image_size = self.current_image.shape[0]
        self.current_position = list(self.start_position)
        self.current_image[self.current_position[1]][self.current_position[0]] = -1
        self.time = 0
        return self.current_image
        
    def change_direction(self, action):
        '''
        {0 : 동작 안함, 1 : 좌측 상단, 2 : 좌측, 3 : 좌측 하단,
        4 : 우측 상단, 5 : 우측, 6 : 우측 하단}
        '''
        degree = None
        if action == 0:
            degree = 0
        elif action == 1:
            degree = -45
        elif action == 2:
            degree = -90
        elif action == 3:
            degree = -135
        elif action == 4:
            degree = 45
        elif action == 5:
            degree = 90
        elif action == 6:
            degree = 135
        
        transformation_matrix = self.generate_transformation_matrix(degree)
        self.current_direction = tuple(np.array(self.current_direction).dot(transformation_matrix).round(0).astype(np.int8))
    
    def generate_transformation_matrix(self, degree):
        degree *= (np.pi / 180.)
        if degree == 90 or degree == -90:
            return np.array([[np.cos(degree), np.sin(degree)],[-np.sin(degree), np.cos(degree)]])
        else:
            return np.array([[np.cos(degree), np.sin(degree)],[-np.sin(degree), np.cos(degree)]])
            
    def step(self, action):
        # change direction
        self.change_direction(action)
        
        # move
        self.old_position = deepcopy(self.current_position)
        self.current_position[0] += self.current_direction[0]
        self.current_position[1] += self.current_direction[1]
        if (0<=self.current_position[0]<self.image_size) and (0<=self.current_position[1]<self.image_size): # not move at the edges
            x, y = self.current_position
            if self.current_image[y][x] in [-1, 1]:
                self.time += 0.5
            elif self.current_image[y][x] == 0:
                self.current_image[y][x] = -1
                self.time += 1
        else:
            self.current_position = self.old_position
            self.time += 1
        
        done = False
        reward = 0
        if self.time >= self.time_end_done or np.sum(self.current_image) == self.fill_up_done:
            done = True
            reward = -self.time
            
        return self.current_image, reward, done
                
    def render(self):
        image = (deepcopy(self.current_image) + 1)/2
        image[self.current_position[1]][self.current_position[0]] = 0.3
        render_img = cv2.resize(image, dsize=(400, 400), interpolation=cv2.INTER_AREA)
        cv2.imshow('Molding...', render_img)
        if cv2.waitKey(25)==ord('q'):
            cv2.destroyAllWindows()

import time
if __name__=="__main__":
    env = env()
    obs = env.reset()
    max_step = 100000
    step = 0
    while step < max_step:
        env.render()
        action = np.random.randint(7)
        _,_,done = env.step(action)
        if done:
            break
    