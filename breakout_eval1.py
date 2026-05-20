import ale_py
import gymnasium as gym
import numpy as np
import datetime as dt
import argparse
import subprocess

from basic import *
from player import *

parser = argparse.ArgumentParser(description='Open AI Gym Evaluator')
parser.add_argument('-r','--render', type=str, default=None, help='Render mode') # human
parser.add_argument('-i','--input', type=str, default=None, help='Input model')
parser.add_argument('-o','--output', type=str, default="model", help='Output model')
parser.add_argument('-mg','--max_games', type=int, default=1000, help='Maximum games') # 1000 # 2500 # 100
parser.add_argument('-ms','--max_steps', type=int, default=18000, help='Maximum steps') # 18000 # according to Igor Pivoarov! (but games are truncated at 108000) 
parser.add_argument('-lm','--learn_mode', type=int, default=2, help='Learn mode (0 - none, 1 - positive only, 2 - positive and negative)')
parser.add_argument('-cs','--context_size', type=int, default=1, help='Context size')
parser.add_argument('-sc','--state_count', type=int, default=2, help='State count threshold')
parser.add_argument('-ss','--state_similarity', type=float, default=0.9, help='State similarity threshold')
parser.add_argument('-tu','--transition_utility', type=int, default=None, help='Transition utility thereshold')
parser.add_argument('-tc','--transition_count', type=int, default=1, help='Transition count threshold')

args = parser.parse_args()

t0 = dt.datetime.now()
grand_t0 = t0
print(f'rev=\"{subprocess.getoutput("git rev-parse HEAD")}\"; time=\"{str(grand_t0)}\"; max_games={args.max_games}; max_steps={args.max_steps}')

# Initialise the environment
#env = gym.make("LunarLander-v3", render_mode="human") # works

# https://gymnasium.farama.org/v0.28.0/environments/atari/breakout/
#env = gym.make('Breakout-v4', render_mode='human') # works
#env = gym.make('BreakoutNoFrameskip-v4', render_mode='human') # works
#env = gym.make('BreakoutNoFrameskip-v4', render_mode='human', obs_type="grayscale") 
env = gym.make('BreakoutNoFrameskip-v4', obs_type="grayscale", render_mode=args.render)

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

#model = None
#model = model_new()
#model=model_read_file(args.model)
model = model_new() if args.input is None else model_read_file(args.input)
print(f"model=\"{args.input}\"; states={len(model['states'])}; games={model['games']}; steps={model['steps']}")

#eval = BreakoutHacky() 
#eval = BreakoutProgrammable(model=model,learn_mode=2,debug=False)
#eval = BreakoutModelDriven(list(range(env.action_space.n)),model=model,learn_mode=args.learn_mode, context_size=args.context_size, args=args, debug=False)
eval = BreakoutModelDrivenNov32025(list(range(env.action_space.n)), model=model, learn_mode=args.learn_mode, context_size=args.context_size, args=args)

scores = []
stepss = []
livess = []
states = []
lapses = []

steps = 0
score = 0
lives = None

#max_steps = 18000 # 18000 # according to Igor Pivoarov! (but games are truncated at 108000) 
#max_games = 1000 # 2500 # 100
game = 0
reward = 0
#action = # env.action_space.sample()
action = 0 # HACK: setting the game!?

# Reset the environment to generate the first observation
#observation, info = env.reset(seed=42)
observation, info = env.reset()
while (game < args.max_games):
    # this is where you would insert your policy
    # if action is None:
    #    action = 2 # env.action_space.sample() # TODO why setting to 0 or 1 crashes on start?
    #else:
    #    action = eval.process_state(observation, reward, action) # pass previous observation, reward (may be negative) and past action (remembered) in 

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if lives == None: # Breakout-specific!!!
        lives = info['lives'] # initial amount of lives
    reward -= (lives - info['lives']) # decrement reward by "lost life" if the life is lost (to pass it to the next process_state)!
    lives = info['lives']

    if reward > 0: # don't subtract lives from rewards!
        score += reward
    elif reward < 0:
        #print(reward,info['lives'],score,scores)
        pass

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated or steps == args.max_steps:
        t1 = dt.datetime.now()
        lapse = t1 - t0
        t0 = t1
        observation, info = env.reset()
        print(f"game={game} cause=\"{'terminated' if terminated else 'truncated' if truncated else f'{args.max_steps}_limit'}\"; " +
              f"score={score}; steps={steps}; lives={lives}; lapse=\"{str(lapse)}\"; states={0 if model is None else len(model['states'])}")
        scores.append(score)
        stepss.append(steps)
        livess.append(lives)
        lapses.append(round(lapse.total_seconds()))
        if not model is None:
            states.append(len(model['states']))
        score = 0
        steps = 0 
        lives = None
        game += 1
        if game == (args.max_games):
            grand_t1 = t1
            total_time = grand_t1 - grand_t0
            print(f'rev=\"{subprocess.getoutput("git rev-parse HEAD")}\"; time=\"{str(grand_t0)}\"; max_games={args.max_games}; max_steps={args.max_steps}')
            print(f"score_avg={round(np.mean(scores),1)}; steps_avg={round(np.mean(stepss),1)}; lives_avg={round(np.mean(livess),1)}; lapse_avg=\"{str(lapse)}\"; time=\"{str(total_time)}\"")
            print('scores =', scores)
            print('stepss =', stepss)
            print('livess =', livess)
            print('lapses =', lapses)
            if not model is None:
                print('states =', states)
                model['games'] += game
                model_write_file(args.output,model)
        action = 1 # HACK: restarting the game !?
        continue # HACK: restarting the game !?

    action = eval.process_state(observation, reward, action) # pass previous observation, reward (may be negative) and past action (remembered) in 

env.close()
