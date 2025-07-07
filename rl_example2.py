#https://medium.com/@alfred.weirich/experiments-with-reinforcement-learning-cff75b7d783c

import argparse
import random
import numpy as np
import os
import pygame

from gymnasium import Env
from gymnasium.spaces import Discrete, Box, MultiDiscrete

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input,  Reshape
from tensorflow.keras.optimizers import Adam


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class maze(Env):
  

    def __init__(self):
        
        
        self.right = 0
        self.left = 1
        self.down = 2
        self.up = 3
        self.stay = 4

        self.actions_directions = ["right", "left", "down", "up", "stay"]

        self.num_rows = 15 
        self.num_cols = 15  
                
        self.state_space = MultiDiscrete([self.num_rows, self.num_cols, 2])
        self.action_space = Discrete(5)

        self.game_board = np.zeros((self.num_rows, self.num_cols))

        self.row = random.randint(0, self.num_rows - 1)
        self.col = random.randint(0, self.num_cols - 1)

        self.num_steps_remaining = 150  # Maximum allowed steps
        self.target=[2, 11] 
        self.reached_target = [self.row,self.col] == self.target
        self.state = np.array([self.row, self.col, self.reached_target])
        
        
        self.allowed = 1  
        self.barrier = 2          
        self.game_board[:,:] = self.barrier          
        self.game_board[0,0:14] = self.allowed 
        self.game_board[0:14, 13] = self.allowed 
        self.game_board[13, 1:14] = self.allowed 
        self.game_board[2:14, 1] = self.allowed 
        self.game_board[2, 1:12] = self.allowed 
        # self.game_board[2:12, 11] = self.allowed 
        # self.game_board[11, 3:12] = self.allowed 
        # self.game_board[4:12, 3] = self.allowed  
        # self.game_board[4, 3:10] = self.allowed  
        # self.game_board[4:10, 9] = self.allowed  
        # self.game_board[9, 5:10] = self.allowed  
        # self.game_board[6:10, 5] = self.allowed  
        # self.game_board[6, 5:8] = self.allowed          
        

        self.model_file = f"./rl_example3/model"
        self.dqn_file = f"./rl_example3/dqn.h5"



    def step(self, action):

        self.num_steps_remaining -= 1 
        done = False
        reward = -200

        if action == self.right and self.col < self.num_cols - 1:
            self.col += 1
            if self.game_board[self.row, self.col] == self.allowed:
                reward = -1
        elif action == self.left and self.col >= 1:
            self.col -= 1
            if self.game_board[self.row, self.col] == self.allowed:
                reward = -1
        elif action == self.down and self.row < self.num_rows - 1:
            self.row += 1
            if self.game_board[self.row, self.col] == self.allowed:
                    reward = -1
        elif action == self.up and self.row >= 1:
            self.row -= 1
            if self.game_board[self.row, self.col] == self.allowed:
                reward = -1            
        elif action == self.stay:
            if self.game_board[self.row, self.col] == self.allowed:
                    reward = -1
        else:
            
            if action == self.right:
                self.col =  self.num_cols - 1
            elif action == self.left:
                self.col =  0
            elif action == self.down:
                self.row = self.num_rows - 1
            elif action == self.up:
                self.row =  0            


                
        if [self.row, self.col] == self.target:
                reward = 200
                self.reached_target=True
        else:
            self.reached_target=False
            

        if self.num_steps_remaining < 0 or self.reached_target==True:
            done = True
  

        self.state[0] = self.row
        self.state[1] = self.col
        self.state[2] = self.reached_target

        info = {}
        return self.state, reward, done,info
    
    
    ############################
    def reset(self):
    
        self.row = random.randint(0, self.num_rows - 1)
        self.col = random.randint(0, self.num_cols - 1)

        self.state[0] = self.row
        self.state[1] = self.col
        self.num_steps_remaining = 150  
             
        if [self.row, self.col] == self.target:
                self.state[2]=True
        else:
            self.state[2]=False

        return self.state  


    def render(self, episode, reward, episode_reward):
       #this function is taken from https://medium.com/@alfred.weirich/experiments-with-reinforcement-learning-cff75b7d783c

        WHITE = (255, 255, 255)  # allowable path
        RED = (255, 0, 0)  # barriers
        BLACK = (0, 0, 0)  # target color
        GREEN = (50, 255, 50)  # current position 

    
        CELL_SIZE = 40  # Size of each grid cell (in pixels)
        ROWS, COLS = self.num_rows, self.num_cols  # Grid dimensions
        WINDOW_WIDTH = COLS * CELL_SIZE  # Total window width
        TEXT_HEIGHT = 35  # Height of the text's
        WINDOW_HEIGHT = ROWS * CELL_SIZE + 3 * TEXT_HEIGHT  # Total window height

        # Initialize Pygame if it hasn't been done yet
        if not hasattr(self, 'screen'):
            pygame.init()  # Initialize the Pygame engine
            self.screen = pygame.display.set_mode(   (WINDOW_WIDTH, WINDOW_HEIGHT))  # Set up the window
            self.clock = pygame.time.Clock()  # Set up the clock for frame rate control
            pygame.font.init()  # Initialize font module
            # Use the default font (size 36)
            self.font = pygame.font.SysFont(None, 36)

        pygame.display.set_caption(f"Episode {episode}")  # Set window caption

        # Handle Pygame events (even though we don't need them right now)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Handle the close window event
                pygame.quit()

        # Clear the screen (fill it with black background)
        self.screen.fill(BLACK)

        # Display text for steps remaining, step reward, and episode score
        steps_text = self.font.render(f"Steps remaining: {self.num_steps_remaining}", True, WHITE)
        reward_text = self.font.render( f"Step reward: {reward}", True, WHITE)
        episode_reward_text = self.font.render(f"Episode reward: {episode_reward}", True, WHITE)

        # Draw the text at the top of the window
        self.screen.blit(steps_text, (10, 10))
        self.screen.blit(reward_text, (10, 40))
        self.screen.blit(episode_reward_text, (10, 70))

        for row in range(ROWS):
            for col in range(COLS):
                # Get the value of the current grid cell
                value = int(self.game_board[row, col])

                if value == self.allowed:
                    color = WHITE  # Valid path
                else:
                    color = RED  # Barrier

                if [row, col] == self.target:
                    color = BLACK  # Load position
 
                if row == self.row and col == self.col:
                        color = GREEN  

                # Draw each cell as a rectangle (offset by TEXT_HEIGHT)
                pygame.draw.rect(self.screen, color, pygame.Rect(
                    col * CELL_SIZE, row * CELL_SIZE + 3 * TEXT_HEIGHT, CELL_SIZE, CELL_SIZE))

                # Optionally, draw grid lines for better visibility
                pygame.draw.rect(self.screen, BLACK, pygame.Rect(
                    col * CELL_SIZE, row * CELL_SIZE + 3 * TEXT_HEIGHT, CELL_SIZE, CELL_SIZE), 1)

        # Update the display to show the rendered grid
        pygame.display.flip()

        # Optionally, control the frame rate for smoother rendering
        self.clock.tick(6)
        
        
        
def run_episode(env, episodes=1, model=None):

    total_score = 0  

    for episode in range(1, episodes + 1):
        print(f"Episode {episode}") 

        state = env.reset().reshape((1, 1, env.state_space.shape[0]))
        done = False 
        episode_reward = 0 

        while not done:

            action = model.predict(state)
            action = np.argmax(action)
            next_state, reward, done, info = env.step(action)
            state = next_state.reshape( (1, 1, env.state_space.shape[0]))
            print(f"State: {state}, Action: {env.actions_directions[action]}, Reward: {reward}")
            episode_reward += reward
            env.render(episode, reward, episode_reward)

        total_score += episode_reward
        print(f"Episode {episode} Score: {episode_reward}")

    print(f"Total Score Across {episodes} Episodes: {total_score}")
    env.close() 
    
    
    
def build_agent(model, actions):

    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=100000, window_length=1)

    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions, 
        nb_steps_warmup=200,  
        target_model_update=1e-2, 
        enable_double_dqn=True  
    )

    return dqn




def build_deepQ_network(in_shape, num_rows, num_cols, num_actions):

    model = Sequential()
    model.add(Input(  shape=(1,) + in_shape     ))
    model.add(Flatten())
    model.add(Dense(num_rows * num_cols*10, activation='sigmoid'   ))
    model.add(Dense(num_actions ,activation='linear'   ))
    
    return model


def main():

    TRAIN = False 
    ROUNDS = 30 
    STEPS = 100  

    parser = argparse.ArgumentParser()
    parser.add_argument('--train',action="store_true")
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--steps', type=int)

    args = parser.parse_args()

    if args.train:
        TRAIN = True
    if args.rounds: 
        ROUNDS = args.rounds  
    if args.steps:
        STEPS = args.steps

    env = maze()
    in_shape = env.state_space.shape  
    num_actions = env.action_space.n 

    if TRAIN: 
        if  os.path.isdir(env.model_file):
            model = tf.keras.models.load_model(env.model_file)  
            dqn = build_agent(model, num_actions)
            dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])
            dqn.load_weights(env.dqn_file)  
  
        else:
            model = build_deepQ_network(in_shape, env.num_rows, env.num_cols, num_actions)
            model.summary()  
            dqn = build_agent(model, num_actions)
            dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])
    
        for fit_round in range(0,ROUNDS):
            print('round is ', fit_round)
            dqn.fit(env, nb_steps=STEPS, visualize=False) 
            model.save(env.model_file)  
            dqn.save_weights(env.dqn_file, overwrite=True)
            scores = dqn.test(env, nb_episodes=20, visualize=False)
            print(scores.history["episode_reward"][-1])
            run_episode(env, model=model, episodes=10)
    else:
        run_episode(env, model=model, episodes=10)
        




if __name__ == "__main__":
    main()