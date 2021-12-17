import numpy as np
from numpy.lib.utils import info
import cv2
from data.main import *
from copy import deepcopy
import matplotlib.pyplot as plt
import pygame
import os

class Env:
    def __init__(self,
                 image_list:list = None,
                 image_size:int = 20,
                 prob_cutout:int = 0.5,
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

        self.current_direction = deepcopy(self.start_direction)
        self.time = 0
        self.total_step = 0
        
        # self.fill_up_done = -(self.image_size**2) + 2*np.sum(self.current_image)
        self.time_end_done = (self.image_size**2)

        # For saving additional information
        self.info = None
        self.trajectory = []
    
    def generate_images(self, img_size, prob_cutout, num_images):
        image_list = []
        for i in range(num_images):
            image_list.append(square_gen(img_size, prob_cutout))
        return image_list
    
    def reset(self):
        # 이미지 초기화
        image_index = np.random.randint(len(self.image_list))
        self.orig_image = deepcopy(self.image_list[image_index])
        self.current_image = deepcopy(self.image_list[image_index])
        self.image_size = self.current_image.shape[0]

        # Done 기준 초기화
        self.fill_up_done = np.sum(self.current_image==0)
        self.time_end_done = (self.image_size**2)

        # Agent 위치 초기화
        self.current_position = list(self.start_position)
        self.current_direction = deepcopy(self.start_direction)
        self.trajectory = []
        x, y = self.current_position
        state = deepcopy(self.current_image)
        state[y][x] = 2
        self.trajectory.append(deepcopy(self.current_position))
        self.current_image[y][x] = -1

        # 기타 정보 초기화
        self.previous_value = 0
        self.time = 0
        self.total_step = 0
        return (state+1)*50
        
    def change_direction(self, action):
        '''
        {0 : 동작 안함, 1 : 좌측 상단, 2 : 좌측, 3 : 좌측 하단,
        4 : 우측 상단, 5 : 우측, 6 : 우측 하단}
        '''
        degree = None
        if action == 0:
            degree = 0
        elif action == 1:
            degree = 45
        elif action == 2:
            degree = 90
        elif action == 3:
            degree = 135
        elif action == 4:
            degree = -45
        elif action == 5:
            degree = -90
        elif action == 6:
            degree = -135
        
        transformation_matrix = self.generate_transformation_matrix(degree)
        self.current_direction = tuple(np.array(self.current_direction).dot(transformation_matrix).round(0).astype(np.int8))
    
    def generate_transformation_matrix(self, degree):
        degree *= (np.pi / 180.)
        return np.array([[np.cos(degree), np.sin(degree)],[-np.sin(degree), np.cos(degree)]])
            
    def step(self, action):
        # change direction
        self.change_direction(action)
        
        # move
        self.old_position = deepcopy(self.current_position)
        
        reward = 0
        done = False

        # move
        self.current_position[0] += self.current_direction[0]
        self.current_position[1] += self.current_direction[1]
        if (0<=self.current_position[0]<self.image_size) and (0<=self.current_position[1]<self.image_size): # not move at the edges
            x, y = self.current_position
            if self.current_image[y][x] == 1:
                self.time += 0.5
                reward = -0.2
                self.time_end_done -= 0.5
            if self.current_image[y][x] == -1:
                self.time += 0.5
                reward = -0.2
                # done = True
            elif self.current_image[y][x] == 0:
                self.current_image[y][x] = -1
                self.time += 1
                # self.time -= 1
                reward = 1
        else:
            self.current_position = self.old_position
            self.time += 1
        
        self.trajectory.append(deepcopy(self.current_position))
        x, y = self.current_position
        state = deepcopy(self.current_image)
        state[y][x] = 2
        
        self.total_step += 1
        # reward = 0
        # if self.time >= self.time_end_done or np.sum(self.current_image) == self.fill_up_done:
        # if self.total_step >= self.time_end_done or np.sum(self.current_image==1) == self.fill_up_done:
        # if np.sum(self.current_image==-1) == self.fill_up_done:
        if self.time_end_done <= 0 or np.sum(self.current_image==-1) == self.fill_up_done:
            done = True
            # reward = -self.time
        
        return (deepcopy(state.astype(np.uint8))+1)*50, reward, done, self.info
                
    def render(self, on_terminal=False, add_comment = ''):
        if on_terminal:
            image = deepcopy(self.current_image)
            image[self.current_position[1]][self.current_position[0]] = 2
            print_on_terminal = ''
            print_on_terminal += '==='*len(image)+'\n'
            for row in image:
                print_on_terminal += ' '.join([f'{int(n):2}' if n!=2 else 'MM' for n in row])+'\n'
            print_on_terminal += '==='*len(image)+'\n'
            print_on_terminal += add_comment
            with open(f'./render/render.log', 'w') as f:
                f.write(print_on_terminal)
            time.sleep(0.03)
        
        else:
            image = (deepcopy(self.current_image) + 1)/2
            image[self.current_position[1]][self.current_position[0]] = 0.3
            render_img = cv2.resize(image, dsize=(400, 400), interpolation=cv2.INTER_AREA)
            cv2.imshow('Molding...', render_img)
            if cv2.waitKey(25)==ord('q'):
                cv2.destroyAllWindows()
    
    def draw_trajectory(self,score):
        image = (deepcopy(self.orig_image) + 1)/2
        plt.imshow(image)
        xs = []
        ys = []
        for pos in self.trajectory:
            xs.append(pos[0])
            ys.append(pos[1])
        plt.plot(xs, ys)
        plt.title(f'trajectory, {score}')
        plt.savefig('./trajectory.png')
        plt.close()

    def heuristic_mode(self, speed=10):
        """Heuristic Mode for Molding

        Args:
            speed (int, optional): Game speed. Defaults to 10.

        Returns:
            (list): Trejectory
        """
        # Initialize the game engine
        pygame.init()
        
        # Define the colors we will use in RGB format
        BLACK= ( 0,  0,  0)
        WHITE= (255,255,255)
        BLUE = ( 0,  0,255)
        GREEN= ( 0,255,  0)
        RED  = (255,  0,  0)
        
        # Set the height and width of the screen
        size  = [400,500]
        screen= pygame.display.set_mode(size)
        font= pygame.font.SysFont("consolas",20)
        background = pygame.surfarray.make_surface(np.zeros(size))
        
        pygame.display.set_caption("Molding")
        
        #Loop until the user clicks the close button.
        done = False
        flag = None
        clock= pygame.time.Clock()

        self.reset()
        
        total_reward = 0
        
        trajectory = []

        while not done:
        
            # This limits the while loop to a max of 10 times per second.
            # Leave this out and we will use all CPU we can.
            clock.tick(speed)
            
            action = 0

            # Main Event Loop
            for event in pygame.event.get():# User did something
                '''
                {0 : 동작 안함, 1 : 좌측 상단, 2 : 좌측, 3 : 좌측 하단,
                4 : 우측 상단, 5 : 우측, 6 : 우측 하단}
                '''
                
                if event.type == pygame.KEYDOWN:# If user release what he pressed.
                    a = event.key
                    if a==ord('q'):
                        action = 1
                    elif a==ord('a'):
                        action = 2
                    elif a==ord('z'):
                        action = 3
                    elif a==ord('e'):
                        action = 4
                    elif a==ord('d'):
                        action = 5
                    elif a==ord('c'):
                        action = 6
                    else:
                        action = 0
                    flag= True
         
                elif event.type == pygame.QUIT: # If user clicked close.
                    done= True                 
                    break                    
            # step
            state, reward, done, info = self.step(action)
            
            # save trajectory
            trajectory.append((state, reward, done, info))
            
            # record reward
            total_reward += reward
            
            # image processing for render
            image = (deepcopy(state.astype(np.uint8))+1)*50
            render_img = cv2.resize(image, dsize=(400, 400), interpolation=cv2.INTER_AREA)
            render = pygame.surfarray.make_surface(render_img)
            
            # info display
            info = font.render(f'time:{self.time_end_done} action:{action} reward:{total_reward:.3f}', True, (255,255,255))
            
            # rendering
            screen.blit(background, (0,0))
            screen.blit(render, (0,100))
            screen.blit(info, (0,10))
            
            # Go ahead and update the screen with what we've drawn.
            # This MUST happen after all the other drawing commands.
            pygame.display.update()
            # pygame.display.flip()
            
            print(f'reward: {total_reward:.3f}', end='\r')
            if done:
                break
        
        # Be IDLE friendly
        pygame.quit()
        return trajectory

import time
if __name__=="__main__":
    env = Env(image_size=10)
    # obs = env.reset()
    # max_step = 100000
    # step = 0
    env.heuristic_mode(5)
    # while step < max_step:
    #     step += 1
    #     env.render()
    #     action = np.random.randint(7)
    #     state,_,done,_ = env.step(action)
        
    #     # if step > 10:
    #     #     for row in state:
    #     #         print(''.join([f'{n:3}' for n in row]))
    #     #     break
        
    #     if done:
    #         env.reset()
    