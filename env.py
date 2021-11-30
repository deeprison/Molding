import numpy as np
from data.main import *

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
        self.image_size = self.image_list[0].shape[0]
                
        # info
        '''
        state_dim : 이미지 크기
        n_action : action 수
        actions : 진행 방향을 기준으로
        {0 : 동작 안함, 1 : 좌측 상단, 2 : 좌측, 3 : 좌측 하단,
        4 : 우측 상단, 5 : 우측, 6 : 우측 하단}
        '''
        self.state_dim = image_list[0].shape
        self.n_action = 7
        self.all_directions = [(0,-1),  #↑
                               (1,-1),  #↗
                               (1,0),   #→
                               (1,1),   #↘
                               (0,1),   #↓
                               (-1,1),  #↙
                               (-1,0),  #←
                               (-1,-1)] #↖
        
        # initialize
        self.current_image = self.image_list[np.random.randint(len(self.img_list))]

        self.start_position = start_position
        if self.start_position == None:
            self.current_position = np.random.choice([(0,0), 
                                                    (0,self.image_size),
                                                    (self.image_size,0),
                                                    (self.image_size, self.image_size)], 1)

        self.start_direction = start_direction
        if self.start_direction == None:
            if self.current_position == (0,0): 
                self.current_direction = np.random.choice([self.all_directions[2], self.all_directions[4]],1)
            elif self.current_position == (0,self.image_size): 
                self.current_direction = np.random.choice([self.all_directions[0], self.all_directions[2]],1)
            elif self.current_position == (self.image_size,0): 
                self.current_direction = np.random.choice([self.all_directions[4], self.all_directions[6]],1)
            elif self.current_position == (self.image_size, self.image_size): 
                self.current_direction = np.random.choice([self.all_directions[0], self.all_directions[6]],1)

        self.start_direction = list(self.start_direction)
        self.time = 0
    
    def generate_imgs(self, img_size, prob_cutout, num_images):
        image_list = []
        for i in range(num_images):
            image_list.append(square_gen(img_size, prob_cutout))
        return image_list
    
    def reset(self):
        image_index = np.random.randint(len(self.img_list))
        self.current_image = self.image_list[image_index]
        self.current_position = self.start_position
        self.time = 0
        
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
        self.current_direction = tuple(np.array(self.current_direction).dot(transformation_matrix))
    
    def generate_transformation_matrix(self, degree):
        if degree == 90 or degree == -90:
            return np.array([[np.cos(degree), np.sin(degree)],[-np.sin(degree), np.cos(degree)]])
        else:
            return np.array([[np.cos(degree), np.sin(degree)],[-np.sin(degree), np.cos(degree)]])/np.cos(degree)
            
    def step(self, action):
        # change direction
        self.change_direction(action)
        
        # move
        self.old_position = self.current_position
        self.current_position[0], self.current_position[1] += self.current_direction[0], self.current_direction[1]
        # not move at the edges
        if self.current_position.any() < 0 or self.current_position.any() > self.image_size:
            self.current_direction = self.old_position
        
    
    def render(self):
        pass