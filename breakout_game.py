import ale_py
import gymnasium as gym

import time
import keyboard


#env = gym.make('ALE/Breakout-v5', render_mode='human', obs_type="ram") 
#env = gym.make('Breakout-v4', render_mode='human', obs_type="ram") 
#env = gym.make('BreakoutNoFrameskip-v4', frameskip = 4, render_mode='human', obs_type="grayscale") 
env = gym.make('BreakoutNoFrameskip-v4', render_mode='human', obs_type="ram") 
observation, info = env.reset()
old_lives = 0
score = 0
while True:
    """
    Value Meaning
    0 NOOP
    1 FIRE new ball
    2 RIGHT
    3 LEFT
    """
    action = 0
    if keyboard.is_pressed('left') or keyboard.is_pressed('a'):
        action = 3
    elif keyboard.is_pressed('right') or keyboard.is_pressed('d'):
        action = 2
    elif keyboard.is_pressed('up') or keyboard.is_pressed('w'):
        action = 1
    elif keyboard.is_pressed('down') or keyboard.is_pressed('s'):
        print('Game Paused. Press any key to continue.') 
        while keyboard.is_pressed('down'):
            time.sleep(0.1)
        keyboard.read_key()
    elif keyboard.is_pressed('esc'):
        env.close()
        break

    observation, reward, terminated, truncated, info = env.step(action)

    if (info['lives'] != old_lives):
        old_lives = info['lives']
        print('lives: ' + str(old_lives))

    if reward != 0: # can be negative or positive
        score += reward
    
    if terminated or truncated:
        print(f'Game Over. Your score is {int(score)}. Press any key to restart.')
        keyboard.read_key()
        observation, info = env.reset()

    #time.sleep(0.02)
    #time.sleep(0.06)