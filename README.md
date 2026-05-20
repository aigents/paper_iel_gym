# Code for paper submission "Experiential Reinforcement Learning Based on State History and Global Feedback" 

## Files

- [README.md](./README.md) - this file
- [LICENSE](./LICENSE) - MIT license file
- [requirements.txt](./requirements.txt) - external dependencies

### Code

- [breakout_hack_simple.py](./breakout_hack_simple.py) - "cheating" Breakout player, knows rules of the game and properties of the game field, wins 860 (out of top 864 points) always if not limited by number of steps per game (for the reference)
- [breakout_hack_simple1.py](./breakout_hack_simple1.py) - same as above "cheating" logic wrapped in BreakoutHacky class (for the reference)
- [breakout_eval2.py](./breakout_eval2.py) - main evaluation script, see script code for command line options
- [player.py](./player.py) - main script with learning and acting functinal in repective classes decribed below    
- [moel_pack.py](./model_pack.py) - utility to analyse and restructure model file, see script code for command line options
- [basic.py](./basic.py) - basic utility functions and operation with model

#### Classes

- ```player.BreakoutHacky``` - "cheating" Breakout player, knows rules of the game and properties of the game field
  - if action = env.action_space.sample() at game (re)start: wins 831/860 (out of top 864 points, varies even with fixed random seed) if not limited by number of steps per game (108000 steps max) or 616/725/732/856 (varies even with fixed random seed) if limited by 18000 steps
  - if action = 1 at game (re)start: wins 860 (out of top 864 points) always (regardless of random seed) if not limited by number of steps per game (108000 steps max) or 732 (regardless of random seed) if limited by 18000 steps
- ```player.BreakoutXXProgrammable``` - "Automated" player which transforms input observations to X coordinates of pixel clouds correspondingg to horizontal positions of the ball and the racket
- ```player.BreakoutProgrammable``` - "Automated" player with extra ability to learn the model states
- ```player.BreakoutModelDriven``` - "Model-based" player - newer version, provides worse performance so far 
- ```player.BreakoutModelDrivenNov32025``` - "Model-based" player - old version, provides the best performance so far 

### Notebooks with Results

The follwing notebooks keep the results of expeiments performed in command line accordingly to the **Instructions** below, with outputs taken from console outputs andd placed in the notebook for visualization and analysis.     

- ```gym_atari_breakout_experiential1.ipynb``` - Results of Phase 1 (Cursory study, Programmable and Learnable Players)
- ```gym_atari_breakout_experiential2.ipynb``` - Results of Phase 2 (Cursory study, Learnable Player)
- ```gym_atari_breakout_experiential3.ipynb``` - Results of Phase 3 (Systematic Study, Learnable Player, 18000 steps/frames limit)  

## Instructions

- Using Python 3.11.13, make sure that have sttandad Python components, ```pip``` and ```vitualenv``` are installed, see https://1cloud.ru/help/linux/ustanovka-jupyter-notebook-na-ubuntu-18-04 
- run ```virtualenv env or python -m venv env```
- run ```. env/bin/activate``` (or ```. ./env/Scripts/activate``` - if under Windows)
- run ```pip install -r requirements.txt```
- run ```python ./breakout_eval2.py``` with specific parameters (non-default parameters are specified in the body of this script)

For instance:
```
% python ./breakout_eval2.py -cs=2 -ss=0.9 -tu=0 -s=41 -mg=20 

A.L.E: Arcade Learning Environment (version 0.11.2+ecc1138)
[Powered by Stella]
rev="fatal: not a git repository (or any of the parent directories): .git"; time="2026-05-20 15:30:16.020604"; max_games=20; max_steps=18000; max_total=100000000; seed=41
model="None"; states=0; games=0; steps=0
learn_mode=2; context_size=2; state_count=2; state_similarity=0.9; transition_utility=0; transition_count=1
Total actions: 4
All possible actions: [0, 1, 2, 3]
game=0 cause="terminated"; score=0; steps=488; lives=0; lapse="0:00:00.413721"; states=331
game=1 cause="terminated"; score=2.0; steps=799; lives=0; lapse="0:00:00.753478"; states=975
game=2 cause="terminated"; score=3.0; steps=855; lives=0; lapse="0:00:01.269089"; states=1263
game=3 cause="terminated"; score=1.0; steps=606; lives=0; lapse="0:00:01.474056"; states=1468
game=4 cause="terminated"; score=3.0; steps=922; lives=0; lapse="0:00:02.801988"; states=1795
game=5 cause="terminated"; score=2.0; steps=739; lives=0; lapse="0:00:02.519709"; states=1959
game=6 cause="terminated"; score=3.0; steps=924; lives=0; lapse="0:00:02.683664"; states=2146
game=7 cause="terminated"; score=3.0; steps=856; lives=0; lapse="0:00:01.625518"; states=2211
game=8 cause="terminated"; score=3.0; steps=851; lives=0; lapse="0:00:04.182191"; states=2403
game=9 cause="terminated"; score=1.0; steps=611; lives=0; lapse="0:00:04.604622"; states=2581
game=10 cause="terminated"; score=4.0; steps=1191; lives=0; lapse="0:00:09.621764"; states=2850
game=11 cause="terminated"; score=3.0; steps=930; lives=0; lapse="0:00:06.610452"; states=3013
game=12 cause="terminated"; score=3.0; steps=915; lives=0; lapse="0:00:05.913610"; states=3153
game=13 cause="terminated"; score=3.0; steps=927; lives=0; lapse="0:00:08.555638"; states=3351
game=14 cause="terminated"; score=2.0; steps=723; lives=0; lapse="0:00:05.035705"; states=3501
game=15 cause="terminated"; score=3.0; steps=920; lives=0; lapse="0:00:08.100738"; states=3702
game=16 cause="terminated"; score=2.0; steps=750; lives=0; lapse="0:00:06.231948"; states=3830
game=17 cause="terminated"; score=4.0; steps=1114; lives=0; lapse="0:00:09.963015"; states=4001
game=18 cause="terminated"; score=4.0; steps=1050; lives=0; lapse="0:00:10.251711"; states=4158
game=19 cause="terminated"; score=3.0; steps=862; lives=0; lapse="0:00:09.782413"; states=4273
rev="fatal: not a git repository (or any of the parent directories): .git"; time="2026-05-20 15:30:16.020604"; max_games=20; max_steps=18000; seed=41
score_avg=2.6; steps_avg=851.6; lives_avg=0.0; lapse_avg="0:00:09.782413"; time="0:01:42.395030"
scores = [0, 2.0, 3.0, 1.0, 3.0, 2.0, 3.0, 3.0, 3.0, 1.0, 4.0, 3.0, 3.0, 3.0, 2.0, 3.0, 2.0, 4.0, 4.0, 3.0]
stepss = [488, 799, 855, 606, 922, 739, 924, 856, 851, 611, 1191, 930, 915, 927, 723, 920, 750, 1114, 1050, 862]
livess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
lapses = [0, 1, 1, 1, 3, 3, 3, 2, 4, 5, 10, 7, 6, 9, 5, 8, 6, 10, 10, 10]
states = [331, 975, 1263, 1468, 1795, 1959, 2146, 2211, 2403, 2581, 2850, 3013, 3153, 3351, 3501, 3702, 3830, 4001, 4158, 4273]
```


