import sys
import random
import numpy as np
from queue import Queue, Full, Empty

memory_size = 30
background_refresh_rate = 10
reactivity = 4
threshold = 1

observations = Queue(maxsize=memory_size) 
epoch = 0


def debug_array2str(a,t):
    return ''.join(['.' if i < t else '█' for i in a])

def print_debug(array2d):
    for a in array2d:
        #print(a)
        print(debug_array2str(a,1))
    #print(len(array2d),len(array2d[0]))

racket_row = None
diff_vert = None
average_array = None 
ball_col_old = None
racket_col_old = None

lives = None
score = 0
scores = []

states = {}
transitions = {}
episode = []
previous_state = None

def count_state(current,previous):
    if not current in states: # count values of transitions
        states[current] = 0
    if previous is None:
        return
    if not previous in transitions:
        transitions[previous] = set()
    transitions[previous].add(current)
    #print(f'Addded transition, total {len(transitions)}')


def process_state(observation, reward, actions, past_action, feelings, debug = True):
    """
    Value Meaning
    0 NOOP
    1 FIRE
    2 RIGHT
    3 LEFT
    """
    global epoch
    global average_array
    global previous_state # TODO cleanup on game restart!?

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
        act = random.choice(actions)
    else:
        diff = np.maximum(np.subtract(observation,average_array),0)
        diff_hor = np.mean(diff, axis=0)
        diff_hor_bin = (diff_hor >= threshold).astype(int)

        if debug:
            #print_debug(observation) # OK - binary map raw
            #print_debug(diff) # OK - binary map of ball and rocket
            #print(diff_vert)
            #print('racket_row',racket_row)
            #print(diff[racket_row])
            #print(np.heaviside(diff_hor,0))
            #print(debug_array2str(diff_hor_bin,1),past_action,feelings,reward)
            if feelings[1] > 0 and False:
                sys.exit()
                #try:
                #    input("Press enter to continue")
                #except SyntaxError:
                #    pass


        state = tuple(list(diff_hor_bin.astype(np.int8)) + list(np.array(past_action + feelings, dtype=np.int8)))
        count_state(state,previous_state)
        episode.append(state)
        previous_state = state

        if reward != 0:
            # propagate "global feedback" value
            for e in episode:
                states[e] += reward # value all states in episode with no temporal decay
            if debug and (reward > 0 or reward < -1):
                print(f'Propagated {reward} to {len(episode)} states')
                print(len(transitions))
            episode.clear()

        #TODO make choice
        chosen_action = None
        if state in transitions:
            options = transitions[state]
            #if debug and len(options) > 0:
            #    print(f'Choosing from {len(options)} states')
            vmax = -1000
            for o in options:
                action_state = o[len(diff_hor_bin):len(diff_hor_bin)+len(past_action)]
                a = np.argmax(action_state) # state action
                v = states[o] # state value
                if v > vmax:
                    vmax = v
                    chosen_action = a
            if debug:
                #print(f'Chosen {chosen_action} from {action_state} among {len(options)} states based on {vmax}')
                pass
                #sys.exit(0)

        act = random.choice(actions) if chosen_action is None else chosen_action

    return act





import ale_py
import gymnasium as gym

# Initialise the environment
#env = gym.make("LunarLander-v3", render_mode="human") # works

# https://gymnasium.farama.org/v0.28.0/environments/atari/breakout/
#env = gym.make('ALE/Breakout-v5', render_mode='human', obs_type="grayscale") 
#env = gym.make('Breakout-v4', render_mode='human', obs_type="grayscale")
#env = gym.make('BreakoutNoFrameskip-v4', frameskip = 4, render_mode='human', obs_type="grayscale") 
env = gym.make('BreakoutNoFrameskip-v4', render_mode='human', obs_type="grayscale") 

#env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array', obs_type="grayscale") 
#env = gym.wrappers.RecordVideo(
#    env,
#    episode_trigger=lambda num: num % 2 == 0,
#    video_folder="saved-video-folder",
#    name_prefix="video",
#)

all_actions = None

# For discrete action spaces (like Atari games)
if hasattr(env.action_space, 'n'):
    print(f"Total actions: {env.action_space.n}")
    all_actions = list(range(env.action_space.n))
    print("All possible actions:", all_actions)

# Get action meanings
if hasattr(env, 'get_action_meanings'):
    action_meanings = env.get_action_meanings()
    print("Action meanings:", action_meanings)
    # Create a mapping of action numbers to their meanings
    for i, meaning in enumerate(action_meanings):
        print(f"Action {i}: {meaning}")

action = None

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(50000):
    # this is where you would insert your policy
    if action is None:
        action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    if lives == None: # Breakout-specific!!!
        lives = info['lives']
    loss = lives - info['lives']
    lives = info['lives']

    if reward != 0: # can be negative or positive
        score += reward
        print(reward,info['lives'],score,scores)

    past_action = [1 if a == action else 0 for a in all_actions] # binary one-hot action vector
    feelings = [1 if reward > 0 else 0, 1 if loss > 0 else 0]
    action = process_state(observation, reward, all_actions, past_action, feelings, False)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        scores.append(score)
        score = 0
        lives = None

print(np.mean(scores))
env.close()
