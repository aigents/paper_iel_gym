import sys
import random
import numpy as np
from queue import Queue, Full, Empty

memory_size = 30
background_refresh_rate = 10
reactivity_base = 4
randomness = 0
 
#observation_top = 0 # Fair
observation_top = 93 # Hack for Breakout - cut ceiling off
#observation_border = 0 # Fair
observation_border = 8 # Hack for Breakout - cut walls off

observations = Queue(maxsize=memory_size) 
epoch = 0


def debug_array2str(a,t):
    return ''.join(['.' if i < t else '█' for i in a])

def print_debug(array2d):
    row = 0
    for a in array2d:
        print(debug_array2str(a,1),row)
        row += 1

#debug = True
racket_row = None
diff_vert = None
average_array = None 
ball_col_old = None
racket_col_old = None

def get_avg_pos(a,t):
    indexes = np.where(a > t)[0]
    #print(indexes)
    return np.mean(indexes) if len(indexes) > 0 else None

def process_state(observation, reward, debug = True):
    """
    Value Meaning
    0 NOOP
    1 FIRE
    2 RIGHT
    3 LEFT
    """
    global epoch
    global average_array

    if observation_top > 0: 
        observation = observation[observation_top:]
    if observation_border > 0: 
        observation = [o[observation_border:-observation_border] for o in observation]

    # accumulate observations 
    if observations.qsize() == memory_size:
        observations.get()
    observations.put((observation, reward))
    epoch += 1

    # update background
    if epoch % background_refresh_rate == 0: 
        observation_maps = [a[0] for a in list(observations.queue)] # grayscale!
        average_array = np.mean(observation_maps, axis=0)
    
    if average_array is None:
        act = 0
    else:
        diff = np.maximum(np.subtract(observation,average_array),0)
        global racket_row
        global diff_vert
        if racket_row is None:
            max = 0
            diff_vert = [int(np.sum(d)) for d in diff] 
            for row in range(len(diff_vert)):
                if diff_vert[row] > max:
                    max = diff_vert[row]
                    racket_row = row
            #print(racket_row)
        diff_ball = diff[0:racket_row]
        #ball_col = np.argmax(np.convolve(np.mean(diff_ball, axis=0), [1,1,1], mode='same'))
        #racket_col = np.argmax(np.convolve(diff[racket_row], [1,1,1], mode='same'))
        ball_col = get_avg_pos(np.mean(diff_ball, axis=0),1)
        racket_col = get_avg_pos(observation[racket_row],1)

        global racket_col_old
        global ball_col_old
        racket_dir = 0 if (racket_col_old is None or racket_col is None) else racket_col - racket_col_old
        ball_dir = 0 if (ball_col_old is None or ball_col is None) else ball_col - ball_col_old
        racket_col_old = racket_col
        ball_col_old = ball_col

        if ball_col is None:
            ball_col_pred = len(observation[racket_row]) / 2 # HACK: assume that ball will get into middle by default 
        else:
            ball_col_pred = ball_col + ball_dir * 2 # be over-predictive, double the ball speed!?
        
        reactivity = random.choice(list(range(reactivity_base-randomness,reactivity_base-randomness+1))) # HACK: randomness preventing dead cycles
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

        if debug and 0:
            print_debug(observation) # OK - binary map raw
            sys.exit()
            #print_debug(diff) # OK - binary map of ball and rocket
            #print(diff_vert)
            #print('racket_row',racket_row)
            #print(diff[racket_row])
            print(ball_col,ball_col_pred,racket_col,act)
            print('===')
            try:
                input("Press enter to continue")
            except SyntaxError:
                pass

        if debug:
            #print(str(np.mean(diff_ball, axis=0)))
            #print(str(diff[racket_row]))
            #print(get_avg_pos(np.mean(diff_ball, axis=0),1))
            #print(get_avg_pos(diff[racket_row],1))
            #print(np.convolve(np.mean(diff_ball, axis=0), [1,1,1], mode='same'))
            #print(np.convolve(diff[racket_row], [1,1,1], mode='same'))
            print(debug_array2str(np.mean(diff_ball, axis=0),1),ball_col,ball_dir,ball_col_pred)
            print(debug_array2str(diff[racket_row],1),racket_col,racket_dir,act)
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

debug_count = 0

action = None

# Reset the environment to generate the first observation
#observation, info = env.reset(seed=42)
observation, info = env.reset()
while (True):
    # this is where you would insert your policy
    if action is None:
        action = 2 # env.action_space.sample() # TODO why setting to 0 or 1 crashes on start?
    else:
        action = process_state(observation, reward)

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
    if terminated or truncated:
        observation, info = env.reset()
        scores.append(score)
        stepss.append(steps)
        livess.append(lives)
        score = 0
        steps = 0 
        lives = None
        print('terminated' if terminated else 'truncated')
        print('scores =', scores)
        print('steps =', stepss)
        print('lives =', livess)
        print('==============')


env.close()
