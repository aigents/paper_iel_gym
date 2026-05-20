#import sys
import numpy as np
from queue import Queue, Full, Empty
from collections import deque

from basic import *

class BreakoutHacky(GymPlayer):
    def __init__(self,model=None,debug=False):
        super().__init__(debug)
        self.model = model
        self.rocket_row = None
        self.diff_vert = None
        self.initial_array = None 
        self.prev_rocket_size = 0

    def process_state(self, raw_observation, reward, previous_action):

        observation = np.ndarray((178,144))
        for i in range(32, len(raw_observation)):
            observation[i-32] = raw_observation[i][8:152]

        if self.initial_array is None:
            self.initial_array = np.copy(observation)
            return 1 # fire ball 
        
        if self.rocket_row is None:
            max = 0
            self.diff_vert = [int(np.sum(d)) for d in observation] 
            for row in range(100, len(self.diff_vert)): # ignore rows with tiles
                if self.diff_vert[row] > max:
                    max = self.diff_vert[row]
                    self.rocket_row = row
            if self.debug:
                print(self.rocket_row) # = 157

        diff = np.maximum(np.subtract(observation,self.initial_array),0)
        diff_ball = diff[0:self.rocket_row]

        # ball size 2, rocket size starts with 16 and gets smaller
        ball_col = np.argmax(np.convolve(np.mean(diff_ball, axis=0), [1,1,1], mode='same')) + (2)/2
        rocket_size = int(np.sum([1 for d in observation[self.rocket_row] if d>0]))
        rocket_col = np.argmax(np.convolve(observation[self.rocket_row], [1,1,1], mode='same')) - 1
        if rocket_col + rocket_size <= 143: # no right wall collision
            self.prev_rocket_size = rocket_size
        rocket_col += (self.prev_rocket_size)/2

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

        if rocket_col < ball_col - 4:
            if rocket_col + (self.prev_rocket_size)/2 > 143:
                act = 0 # NOOP - right wall collision
            else:    
                act = 2 # RIGHT
        elif rocket_col > ball_col + 4:
            act = 3 # RIGHT
        else:
            act = 0 # NOOP

        return act




class BreakoutXXProgrammable(GymPlayer):
    def __init__(self,width,debug=False):
        super().__init__(debug)
        self.reactivity_base = 4
        self.randomness = 0

        self.ball_col_old = None
        self.ball_dir = 0 # ball direction and speed
        self.racket_col_old = None
        self.width = width

        self.started = False
        self.prev_racket_size = 0


    def process_state_complex(self, observation, reward): # Original Anton's version
        (racket_col, ball_col) = observation
        if racket_col is None:
            act = 0
        else:
            racket_dir = 0 if (self.racket_col_old is None or racket_col is None) else racket_col - self.racket_col_old # not used, for debugging 
            self.racket_col_old = racket_col

            if ball_col is None: # ball is not "visible"
                #ball_col_pred = len(observation[self.racket_row]) / 2 # HACK: assume that ball will get into middle by default
                ball_col_pred = random.choice(list(range(self.width))) # HACK: guess where the ball is randomly
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

        return act
    

    def process_state(self, observation, reward, racket_size): # Latest Vladimir's version adopted by Anton
        (racket_col, ball_col) = observation
        if not self.started:
            self.started = True
            return 1

        #print(self.width - 1,self.width /2)

        if racket_col + racket_size/2 <= (self.width - 1): # 143 # no right wall collision
            self.prev_racket_size = racket_size

        # check if ball is in game
        if ball_col == INT_NONE:
            if random.choice([True, False]): # HACK: to fire the ball enventually 
                return 1
            # return rocket to the center of the screen
            if racket_col < self.width / 2: # == 72
                return 2 # RIGHT
            elif racket_col > self.width /2: # == 72
                return 3 # LEFT
            else:
                return 1 # fire new ball # TODO: why do we never get here?
        if racket_col < ball_col - 4:
            if racket_col + self.prev_racket_size/2 > (self.width - 1): # 143
                act = 0 # NOOP - right wall collision
            else:    
                act = 2 # RIGHT
        elif racket_col > ball_col + 4:
            act = 3 # RIGHT
        else:
            act = 0 # NOOP
        return act


class BreakoutProgrammable(GymPlayer):

    def __init__(self,model=None,learn_mode=0,context_size=1,debug=False):
        super().__init__(debug)
        self.memory_size = 30
        self.background_refresh_rate = 10
        
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

        self.diff = None # may be not used

        self.eval = None

        self.model = model
        if context_size > 1:
            model_set_context_size(self.model,context_size)
        self.learn_mode = learn_mode
        self.context_size = context_size
        self.context_states = deque(maxlen=context_size) # all very latest states
        self.states = [] # all states in emotionally reinforced context

    def process_observation(self,observation,reward,previous_action):
        if self.observation_top > 0: 
            observation = observation[self.observation_top:]
        if self.observation_border > 0: 
            observation = [o[self.observation_border:-self.observation_border] for o in observation]
        # accumulate observations in rolling window
        if self.observations.qsize() == self.memory_size:
            self.observations.get()
        self.observations.put((observation, reward))
        # update background
        self.epoch += 1
        if self.epoch % self.background_refresh_rate == 0: 
            self.observation_maps = [a[0] for a in list(self.observations.queue)] # grayscale!
            self.average_array = np.mean(self.observation_maps, axis=0)
        return observation

    def racket_ball_x(self,observation):
        if self.racket_row is None:
            if self.average_array is None:
                return (INT_NONE, INT_NONE)
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
        return (INT_NONE if racket_x is None else int(round(racket_x)), INT_NONE if ball_x is None else int(round(ball_x)))

    def learn_model(self,state,reward):
        if not self.model is None and self.learn_mode != 0:
            if reward != 0:
                if self.learn_mode == 1:
                    feedback = reward if reward > 0 else 0 # positive only
                else:
                    feedback = reward # positive or negative
                #print(f"learn_mode={self.learn_mode}; context_size={self.context_size}; state_count={self.state_count_threshold}; state_similarity={self.state_similarity_threshold}; transition_utility={self.transition_utility_thereshold}; transition_count={self.transition_count_threshold}")
                model_add_states_contexts(self.model,self.states,feedback)
                #print(len(self.model['states']),len(self.model['contexts'][2]))
                self.states.clear() # clear the states including the rewarded one to start over with new state and new action on it
            self.states.append(state)


    def process_state(self, observation, reward, previous_action):
        """
        Value Meaning
        0 NOOP
        1 FIRE
        2 RIGHT
        3 LEFT
        """
        observation = self.process_observation(observation,reward,previous_action)

        # find racket & ball X
        (racket_col, ball_col) = self.racket_ball_x(observation)
        current_state = (previous_action,)+(1 if reward > 0 else 0,1 if reward < 0 else 0)+(racket_col, ball_col)
        #TODO aggregate states based on context_size
        if self.context_size == 1:
            state = current_state
        else:
            pass #TODO states

        self.learn_model(state,reward)

        if self.eval is None: # Lazy init
            self.eval = BreakoutXXProgrammable(len(observation[self.racket_row]))

        #act = self.eval.process_state((racket_col, ball_col), reward)
        act = self.eval.process_state((racket_col, ball_col), reward, racket_size = np.sum([1 for d in observation[self.racket_row] if d>0]))

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
                print(debug_array2str(np.mean(observation[0:self.racket_row], axis=0),1),ball_col,self.ball_dir)
                print(debug_array2str(observation[self.racket_row],1),self.racket_row,racket_col,act)

        return act


# New code after Nov 3 2025 - performs worse TODO explore why

class BreakoutModelDriven(BreakoutProgrammable): # State-based History-aware Artificial Reinforcement Intelligent Kernel (SHARIK)

    def __init__(self,actions,model=None,learn_mode=0,context_size=2,args=None,debug=False):
        super().__init__(model,learn_mode,context_size,debug)
        self.actions = actions
        self.state_count_threshold = 2 if args is None else args.state_count
        self.state_similarity_threshold = 0.9 if args is None else args.state_similarity
        self.transition_utility_thereshold = 0 if args is None else args.transition_utility
        self.transition_count_threshold = 1 if args is None else args.transition_count
        #print(type(self.learn_mode),type(self.context_size),type(self.state_count_threshold),type(self.state_similarity_threshold),type(self.transition_utility_thereshold),type(self.transition_count_threshold))
        print(f"learn_mode={self.learn_mode}; context_size={self.context_size}; state_count={self.state_count_threshold}; state_similarity={self.state_similarity_threshold}; transition_utility={self.transition_utility_thereshold}; transition_count={self.transition_count_threshold}")

    def process_state(self, observation, reward, previous_action):
        observation = self.process_observation(observation,reward,previous_action)

        # find racket & ball X
        (racket_col, ball_col) = self.racket_ball_x(observation)
        state = (previous_action,)+(1 if reward > 0 else 0,1 if reward < 0 else 0)+(racket_col, ball_col)

        self.learn_model(state,reward)

        found = None
        if self.context_size > 1 and len(self.states) > 1: #TODO make other than 2
            context = sum(self.states[-2:],())
            contexts = self.model['contexts'][2] #TODO make other than 2
            try:
                found = contexts[context]
                match = 'exact2'
            except KeyError:
                if self.state_similarity_threshold < 1.0:
                    found = find_similar(contexts,context, self.state_count_threshold, self.state_similarity_threshold )
                match = 'similar2'
        if found is None:
            states = self.model['states']
            try:
                found = states[state]
                match = 'exact1'
            except KeyError:
                if self.state_similarity_threshold < 1.0:
                    found = find_similar(states,state, self.state_count_threshold, self.state_similarity_threshold)
                match = 'similar1'

        if not found is None:
            (utility,count,transitions) = found
            #print('found',match,state,'=>',found,'=',len(transitions))
            # new code TODO: fix and test!!!
            return find_useful_action(self.actions,transitions, transition_utility_thereshold=0, transition_count_threshold=1)
            # old code:
            best = find_useful(transitions,transition_utility_thereshold=0,transition_count_threshold=1)
            if not best is None:
                if self.debugging():
                    print('found',match,utility,count,len(transitions),best[0] if not best is None else '-')
                return best[0]
    
        #print("found none")
        return random.choice(self.actions)


## Old code from Nov 3 2025 - performs better now TODO explore why 

def find_similarNov32025_with_rand(states,state,count_threshold,similarity_threshold):
    #print(f'find_similarNov32025_with_rand count_threshold={count_threshold} similarity_threshold={similarity_threshold}')
    max_sim = 0
    bests = []
    for s, utility_count in states.items():
        if utility_count[1] < count_threshold: # disregard rare evidence
            continue
        sim = cosine_similarity(s,state)
        if sim < similarity_threshold:
            continue
        if max_sim < sim:
            bests.clear()
            max_sim = sim
            bests.append(s)
        elif max_sim == sim:
            bests.append(s)
    best = bests[0] if len(bests) == 1 else random.choice(bests) if len(bests) > 1 else None
    return states[best] if not best is None else None

def find_usefulNov32025(transitions,utility_thereshold,count_threshold,counted_utility=False):
    #print(f'find_usefulNov32025 utility_thereshold={utility_thereshold} count_threshold={count_threshold}')
    max_utility = None
    max_count = 0
    best = None
    for s, utility_count in transitions.items():
        utility, count = utility_count
        if (not utility_thereshold is None) and utility < utility_thereshold: # disregard low utility
            continue
        if count < count_threshold: # disregard rare evidence
            continue
        if counted_utility:
            utility *= count
        if max_utility is None or max_utility < utility:
            max_utility = utility
            max_count = count
            best = s
    if not best is None:
        #print('found',max_utility,max_count,len(transitions),best[0] if not best is None else '-')
        return best


# TODO remove or merge later
class BreakoutModelDrivenNov32025(BreakoutModelDriven):

    def __init__(self,actions,model=None,learn_mode=0,context_size=1,args=None,state_reward=True,encode_action=False,counted_utility=False,debug=False):
        super().__init__(model=model,actions=actions,learn_mode=learn_mode,context_size=context_size,args=args,debug=debug)
        self.state_reward = state_reward
        self.encode_action = encode_action
        self.counted_utility = counted_utility

    def process_state(self, observation, reward, previous_action):
        observation = self.process_observation(observation,reward,previous_action)

        # find racket & ball X
        (racket_col, ball_col) = self.racket_ball_x(observation)
        state_action = one_hot(previous_action,len(self.actions)) if self.encode_action else (previous_action,)
        if self.state_reward:
            state = state_action+(1 if reward > 0 else 0,1 if reward < 0 else 0)+(racket_col, ball_col)
        else:
            state = state_action+(racket_col, ball_col)

        if not self.model is None and self.learn_mode != 0:
            if reward != 0:
                if self.learn_mode == 1:
                    feedback = reward if reward > 0 else 0 # positive only
                else:
                    feedback = reward # positive or negative
                #print(f"learn_mode={self.learn_mode}; context_size={self.context_size}; state_count={self.state_count_threshold}; state_similarity={self.state_similarity_threshold}; transition_utility={self.transition_utility_thereshold}; transition_count={self.transition_count_threshold}")
                model_add_states_contexts(self.model,self.states,feedback)
                #print(len(self.model['states']),len(self.model['contexts'][2]))
                self.states.clear() # clear the states including the rewarded one to start over with new state and new action on it
            self.states.append(state)

        found = None
        #TODO compact this code with cs going down from self.context_size to 1
        if self.context_size > 2 and len(self.states) > 2: #TODO make other than 3
            context = sum(self.states[-3:],())
            contexts = self.model['contexts'][3] #TODO make other than 3
            try:
                found = contexts[context]
                match = 'exact3'
            except KeyError:
                found = find_similarNov32025_with_rand(contexts,context, self.state_count_threshold, self.state_similarity_threshold )
                match = 'similar3'
        if found is None and self.context_size > 1 and len(self.states) > 1: #TODO make other than 2
            context = sum(self.states[-2:],())
            contexts = self.model['contexts'][2] #TODO make other than 2
            try:
                found = contexts[context]
                match = 'exact2'
            except KeyError:
                found = find_similarNov32025_with_rand(contexts,context, self.state_count_threshold, self.state_similarity_threshold )
                match = 'similar2'
        if found is None:
            states = self.model['states']
            try:
                found = states[state]
                match = 'exact1'
            except KeyError:
                found = find_similarNov32025_with_rand(states,state, self.state_count_threshold, self.state_similarity_threshold)
                match = 'similar1'

        #if found:
        #    print(len(state),match)

        if not found is None:
            (utility,count,transitions) = found
            best = find_usefulNov32025(transitions, self.transition_utility_thereshold, self.transition_count_threshold, counted_utility=self.counted_utility)
            #print('found',match,state,'=>',found,'=',len(transitions))
            #print(str([(transitions[t][0],transitions[t][1],t[0]) for t in transitions]))
            #print(best)
            if not best is None:
                #print('found',match,utility,count,len(transitions),best[0] if not best is None else '-')
                action = one_hot_idx(best[:len(self.actions)]) if self.encode_action else best[0]
                return action

        #print("found none")
        return random.choice(self.actions)