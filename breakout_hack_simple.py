import sys
import numpy as np

debug = False

def debug_array2str(a):
    return ''.join(['.' if i == 0 else '█' for i in a])

def print_debug(array2d):
    for a in array2d:
        print(debug_array2str(a))

rocket_row = None
diff_vert = None
initial_array = None 
prev_observation = None
prev_rocket_size = 0
#prev_action = 1

score = 0
scores = []

def process_state(observation, reward):
    """
    Value Meaning
    0 NOOP
    1 FIRE new ball
    2 RIGHT
    3 LEFT
    """
    global debug
    global initial_array
    global rocket_row
    global diff_vert
    global prev_rocket_size
    #global prev_action

    if initial_array is None:
        initial_array = np.copy(observation)
        return 1 # fire ball 
    
    if rocket_row is None:
        max = 0
        diff_vert = [int(np.sum(d)) for d in observation] 
        for row in range(100, len(diff_vert)): # ignore rows with tiles
            if diff_vert[row] > max:
                max = diff_vert[row]
                rocket_row = row
        if debug:
            print(rocket_row) # = 157

    diff = np.maximum(np.subtract(observation,initial_array),0)
    diff_ball = diff[0:rocket_row]

    # ball size 2, rocket size starts with 16 and gets smaller
    ball_col = np.argmax(np.convolve(np.mean(diff_ball, axis=0), [1,1,1], mode='same')) + (2)/2
    rocket_size = int(np.sum([1 for d in observation[rocket_row] if d>0]))
    rocket_col = np.argmax(np.convolve(observation[rocket_row], [1,1,1], mode='same')) - 1
    if rocket_col + rocket_size <= 143: # no right wall collision
        prev_rocket_size = rocket_size
    rocket_col += (prev_rocket_size)/2

    # check if ball is in game
    if int(np.sum([int(np.sum(d)) for d in diff_ball])) == 0:
        # return rocket to the center of the screen
        if rocket_col < 72:
            return 2 # RIGHT
        elif rocket_col > 72:
            return 3 # LEFT
        else:
            #prev_action = 1
            return 1 # fire new ball

    # I tried to reduce oscillation, but it made lower high score
    #if rocket_col < ball_col - 4:
    #    if rocket_col + (prev_rocket_size)/2 > 143:
    #        act = 0 # NOOP - right wall collision
    #        prev_action = 0
    #    else:    
    #        if prev_action == 3:
    #            act = 0 # NOOP - reduce oscillation
    #            prev_action = 2
    #        else:
    #            act = 2 # RIGHT
    #            prev_action = 2
    #elif rocket_col > ball_col + 4:
    #    if prev_action == 2:
    #        act = 0 # NOOP - reduce oscillation
    #        prev_action = 3
    #    else:
    #        act = 3 # RIGHT
    #        prev_action = 3
    #else:
    #    act = 0 # NOOP
    #    prev_action = 0

    if rocket_col < ball_col - 4:
        if rocket_col + (prev_rocket_size)/2 > 143:
            act = 0 # NOOP - right wall collision
        else:    
            act = 2 # RIGHT
    elif rocket_col > ball_col + 4:
        act = 3 # RIGHT
    else:
        act = 0 # NOOP

    if debug and rocket_col == -1:
        #print_debug(observation) # OK - binary map raw
        print_debug(diff) # OK - binary map of ball and rocket
        print(diff_vert)
        print('rocket_row',rocket_row)
        print(diff[rocket_row])
        print(ball_col,rocket_col,act)
        print('===')
        try:
            input("Press enter to continue")
        except SyntaxError:
            pass

    if debug: 
        print(rocket_row, ball_col,rocket_col,act)

        print(prev_rocket_size)
        print_debug(observation)
        print('=+=')
        try:
            input("Press enter to continue")
        except SyntaxError:
            pass

    return act





import ale_py
import gymnasium as gym

# https://gymnasium.farama.org/v0.28.0/environments/atari/breakout/
#env = gym.make('ALE/Breakout-v5', render_mode='human', obs_type="grayscale") 
#env = gym.make('Breakout-v4', render_mode='human', obs_type="grayscale")
#env = gym.make('BreakoutNoFrameskip-v4', frameskip = 4, render_mode='human', obs_type="grayscale") 
env = gym.make('BreakoutNoFrameskip-v4', render_mode='human', obs_type="grayscale") 
#env = gym.make('BreakoutNoFrameskip-v4', obs_type="grayscale") 

#env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array', obs_type="grayscale") 
#env = gym.wrappers.RecordVideo(
#    env,
#    episode_trigger=lambda num: num % 1 == 0,
#    video_folder="saved-video-folder",
#    name_prefix="video",
#)

# get initial state of the game before firing the ball
action = 0

# Reset the environment to generate the first observation
observation, info = env.reset()
for _ in range(140000):
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    raw_observation, reward, terminated, truncated, info = env.step(action)

    observation = np.ndarray((178,144))
    for i in range(32, len(raw_observation)):
        observation[i-32] = raw_observation[i][8:152]

    if reward != 0: # can be negative or positive
        score += reward
        if debug:
            print(reward,info['lives'],score,scores)

    # If the episode has ended reset game and fire ball
    if terminated or truncated:
        observation, info = env.reset()
        scores.append(score)
        score = 0
        action = 1
        print(scores)
        continue

    action = process_state(observation, reward)

print(scores)
env.close()
