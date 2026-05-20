import sys
import numpy as np

from basic import *
from player import *


score = 0
scores = []

eval = BreakoutHacky()


import ale_py
import gymnasium as gym

# https://gymnasium.farama.org/v0.28.0/environments/atari/breakout/
#env = gym.make('ALE/Breakout-v5', render_mode='human', obs_type="grayscale") 
#env = gym.make('Breakout-v4', render_mode='human', obs_type="grayscale")
#env = gym.make('BreakoutNoFrameskip-v4', frameskip = 4, render_mode='human', obs_type="grayscale") 
#env = gym.make('BreakoutNoFrameskip-v4', render_mode='human', obs_type="grayscale") 
env = gym.make('BreakoutNoFrameskip-v4', obs_type="grayscale") 

#env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array', obs_type="grayscale") 
#env = gym.wrappers.RecordVideo(
#    env,
#    episode_trigger=lambda num: num % 1 == 0,
#    video_folder="saved-video-folder",
#    name_prefix="video",
#)

# get initial state of the game before firing the ball
action = 0 # HACK: setting the game!?

# Reset the environment to generate the first observation
observation, info = env.reset()
for _ in range(140000):
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    if reward != 0: # can be negative or positive
        score += reward
        if eval.debug:
            print(reward,info['lives'],score,scores)

    # If the episode has ended reset game and fire ball
    if terminated or truncated:
        observation, info = env.reset()
        scores.append(score)
        score = 0
        print(f"terminated={terminated}, truncated={truncated}",scores)
        action = 1 # HACK: restarting the game !?
        continue # HACK: restarting the game !?

    action = eval.process_state(observation, reward, action)

print(scores)
env.close()
