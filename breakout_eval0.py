import sys
import random
import numpy as np
from queue import Queue, Full, Empty

def debug_array2str(a,t):
    return ''.join(['.' if i < t else '█' for i in a])

def print_debug(array2d):
    row = 0
    for a in array2d:
        print(debug_array2str(a,1),row)
        row += 1

def get_avg_pos(a,t):
    indexes = np.where(a > t)[0]
    #print(indexes)
    return np.mean(indexes) if len(indexes) > 0 else None

class BreakoutEvaluatorProgrammable:

    def __init__(self,debug):
        self.debug = debug
        self.memory_size = 30
        self.background_refresh_rate = 10
        self.reactivity_base = 4
        self.randomness = 0
        
        #self.observation_top = 0 # Fair
        self.observation_top = 93 # Hack for Breakout - cut ceiling off
        #self.observation_border = 0 # Fair
        self.observation_border = 8 # Hack for Breakout - cut walls off

        self.observations = Queue(maxsize=self.memory_size) 
        self.epoch = 0

        self.racket_row = 96 # None # 96 (with top already cut!?)
        self.diff_vert = None
        self.average_array = None 
        self.ball_col_old = None
        self.ball_dir = 0 # ball direction and speed
        self.racket_col_old = None

        self.diff = None # maay be not used


    def racket_ball_x(self,observation):
        if self.observation_top > 0: 
            observation = observation[self.observation_top:]
        if self.observation_border > 0: 
            observation = [o[self.observation_border:-self.observation_border] for o in observation]

        # accumulate observations in rolling window
        if self.observations.qsize() == self.memory_size:
            self.observations.get()
        self.observations.put((observation, reward))

        # update background
        if self.epoch % self.background_refresh_rate == 0: 
            self.observation_maps = [a[0] for a in list(self.observations.queue)] # grayscale!
            self.average_array = np.mean(self.observation_maps, axis=0)

        if self.racket_row is None:

            if self.average_array is None:
                return (None, None)
            
            self.diff = np.maximum(np.subtract(observation,self.average_array),0)
            max = 0
            diff_vert = [int(np.sum(d)) for d in self.diff] 
            for row in range(len(diff_vert)):
                if diff_vert[row] > max:
                    max = diff_vert[row]
                    self.racket_row = row

        #racket_col = np.argmax(np.convolve(diff[racket_row], [1,1,1], mode='same'))
        racket_x = get_avg_pos(observation[self.racket_row],1)

        #ball_col = np.argmax(np.convolve(np.mean(diff_ball, axis=0), [1,1,1], mode='same'))
        ball_hor = observation[0:self.racket_row]
        ball_x = get_avg_pos(np.mean(ball_hor, axis=0),1)

        return (racket_x, ball_x)
    

    def process_state(self, observation, reward, debug = False):
        """
        Value Meaning
        0 NOOP
        1 FIRE
        2 RIGHT
        3 LEFT
        """

        self.epoch += 1

        # find racket & ball X
        (racket_col, ball_col) = self.racket_ball_x(observation)
        
        if racket_col is None:
            act = 0
        else:
            racket_dir = 0 if (self.racket_col_old is None or racket_col is None) else racket_col - self.racket_col_old # not used, for debugging 
            self.racket_col_old = racket_col

            if ball_col is None: # ball is not "visible"
                #ball_col_pred = len(observation[self.racket_row]) / 2 # HACK: assume that ball will get into middle by default
                ball_col_pred = random.choice(list(range(len(observation[self.racket_row])))) # HACK: guess where the ball is randomly
            else:
                self.ball_dir = 0 if (self.ball_col_old is None) else ball_col - self.ball_col_old
                ball_col_pred = ball_col + self.ball_dir * 2 # be over-predictive, double the ball speed!?
    
            self.ball_col_old = ball_col

            reactivity = random.choice(list(range(self.reactivity_base-self.randomness,self.reactivity_base+self.randomness+1))) # HACK: randomness preventing dead cycles
            
            assert(not racket_col is None)
            assert(not ball_col_pred is None)
            if ball_col is None:
                act = 1  
            elif racket_col - ball_col_pred < -reactivity:
                act = 2 # RIGHT
            elif racket_col - ball_col_pred > reactivity:
                act = 3 # LEFT
            else: # TODO FIRE if ball is NOT visible, otherwise 0 (NOOP) !!!
                act = 1 # FIRE 

            if self.debug and 0:
                try:
                    input("Press enter to continue")
                except SyntaxError:
                    pass

            if self.debug:
                #print(str(np.mean(diff_ball, axis=0)))
                #print(str(diff[racket_row]))
                #print(get_avg_pos(np.mean(diff_ball, axis=0),1))
                #print(get_avg_pos(diff[racket_row],1))
                #print(np.convolve(np.mean(diff_ball, axis=0), [1,1,1], mode='same'))
                #print(np.convolve(diff[racket_row], [1,1,1], mode='same'))
                print(debug_array2str(np.mean(observation[self.observation_top:self.observation_top+self.racket_row], axis=0),1),ball_col,self.ball_dir,ball_col_pred)
                print(debug_array2str(observation[self.observation_top+self.racket_row],1),self.racket_row,racket_col,racket_dir,act)
                pass

        return act

import ale_py
import gymnasium as gym

# Initialise the environment
#env = gym.make("LunarLander-v3", render_mode="human") # works

# https://gymnasium.farama.org/v0.28.0/environments/atari/breakout/
#env = gym.make('Breakout-v4', render_mode='human') # works
#env = gym.make('BreakoutNoFrameskip-v4', render_mode='human') # works
env = gym.make('BreakoutNoFrameskip-v4', render_mode='human', obs_type="grayscale") 
#env = gym.make('BreakoutNoFrameskip-v4', obs_type="grayscale")

eval = BreakoutEvaluatorProgrammable(debug=True) 

scores = []
stepss = []
livess = []

steps = 0
score = 0
lives = None

# For discrete action spaces (like Atari games)
if hasattr(env.action_space, 'n'):
    print(f"Total actions: {env.action_space.n}")
    print("All possible actions:", list(range(env.action_space.n)))

# Get action meanings
if hasattr(env, 'get_action_meanings'):
    action_meanings = env.get_action_meanings()
    print("Action meanings:", action_meanings)
    # Create a mapping of action numbers to their meanings
    for i, meaning in enumerate(action_meanings):
        print(f"Action {i}: {meaning}")

max_steps = 18000 # according to Igor Pivoarov!

action = None

# Reset the environment to generate the first observation
#observation, info = env.reset(seed=42)
observation, info = env.reset()
while (True):
    # this is where you would insert your policy
    if action is None:
        action = 2 # env.action_space.sample() # TODO why setting to 0 or 1 crashes on start?
    else:
        action = eval.process_state(observation, reward)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if lives == None: # Breakout-specific!!!
        lives = info['lives'] # initial amount of lives
    reward -= (lives - info['lives']) # decrement reward by "lost life", if the life is lost, according to Igor Pivovarov
    lives = info['lives']

    if reward != 0:
        score += reward
        if reward < 0:
            print(reward,info['lives'],score,scores)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated or steps == max_steps:
        observation, info = env.reset()
        scores.append(score)
        stepss.append(steps)
        livess.append(lives)
        score = 0
        steps = 0 
        lives = None
        print('terminated' if terminated else 'truncated' if truncated else '18000 steps limit')
        print('scores =', scores)
        print('steps =', stepss)
        print('lives =', livess)
        print('==============')


env.close()
