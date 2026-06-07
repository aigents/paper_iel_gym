import os, sys
import numpy as np
import pickle
import random
import math

# Util

INT_NONE = -sys.maxsize - 1

def debug_array2str(a,t):
    return ''.join(['.' if i < t else '█' for i in a])

def print_debug(array2d):
    row = 0
    for a in array2d:
        print(debug_array2str(a,1),row)
        row += 1

def one_hot(val,size):
    if val < 0 or val >= size:
        raise IndexError
    lst = [0] * size
    lst[val] = 1
    return tuple(lst)
assert(str(one_hot(0,4))=='(1, 0, 0, 0)')
assert(str(one_hot(1,4))=='(0, 1, 0, 0)')
assert(str(one_hot(2,4))=='(0, 0, 1, 0)')
assert(str(one_hot(3,4))=='(0, 0, 0, 1)')
#print(str(one_hot(4,4))) #IndexError

def one_hot_idx(lst):
    for index, value in enumerate(lst):
        if value != 0:
            return index
    return -1
assert(one_hot_idx((0,0,0,0))==-1)
assert(one_hot_idx((1,0,0,0))==0)
assert(one_hot_idx((0,1,0,0))==1)
assert(one_hot_idx((0,0,1,0))==2)
assert(one_hot_idx((0,0,0,1))==3)


def get_avg_pos(a,t):
    indexes = np.where(a > t)[0]
    return np.mean(indexes) if len(indexes) > 0 else None


# Basic

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


def max_corner_distance(dims):
    """
    Compute the maximum Euclidean distance (corner-to-corner) for a space
    where each dimension i is bounded in [0, M_i].
    Parameters:
        dims : array-like
            Maximum values for each dimension.
    Returns:
        float : Maximum Euclidean distance.
    """
    #return np.sqrt(np.sum(np.array(dims)**2))
    return np.linalg.norm(dims)

assert(round(max_corner_distance(([10,10])))==14)
# state = (previous_action,)+(1 if reward > 0 else 0,1 if reward < 0 else 0)+(racket_col, ball_col)
assert(round(math.ceil(max_corner_distance([3,1,1,178,178])))==252)


def max_corner_distance_min_max(minmax):
    """
    Compute the maximum Euclidean distance (corner-to-corner) for a space
    where each dimension i is bounded in [Min_i, Max_i].
    Parameters:
        minmax : array-like
            Min and Max values (lists) for each dimension.
    Returns:
        float : Maximum Euclidean distance.
    """
    squared_sum = 0.0
    for mi, ma in minmax:
        assert(mi < ma)
        diff = ma - mi
        squared_sum += diff * diff
    #TODO use np.linalg.norm for sqrt(sum(x**2))
    return np.sqrt(squared_sum)

assert(round(max_corner_distance_min_max(([0,10],[0,10])))==14)
assert(round(max_corner_distance_min_max(([-10,10],[-10,10])))==28)
    

def norm_euclidean_distance(a,b,dims,dist_max):
    """
    Euclidean distance between two vectors noralized by the longest vector, assuming both vectores are in the same  
    """
    #d = np.linalg.norm(np.array(a) - np.array(b)) / dist_max
    c = [(ai-bi if (ai !=INT_NONE and bi != INT_NONE) else di) for ai, bi, di in zip(a,b,dims)]
    if dist_max is None:
        dist_max = max_corner_distance(dims)
    d = np.linalg.norm(c) / dist_max
    if not (d >= 0 and d <=1.0):
        print(a)
        print(b)
        print(c)
        print(d,np.linalg.norm(c),dist_max)
    assert(d >= 0 and d <=1.0) # to make sure it is in range
    return d


def norm_similarity(a,b,method='1/(1+d)',dims=None,dist_max=None):
    if method == 'cos':
        return cosine_similarity(a,b) 
    if method == '1-dnorm': # equivalent to '1-d/max'
        a = [(ai / di if ai !=INT_NONE else ai) for ai, di in zip(a,dims)]
        b = [(bi / di if bi !=INT_NONE else bi) for bi, di in zip(b,dims)]
        c = [(ai-bi if (ai !=INT_NONE and bi != INT_NONE) else 1) for ai, bi in zip(a,b)]
        d = np.linalg.norm(c) / math.sqrt(len(dims)) # sqrt(sum(1**2))
        if not (d >= 0 and d <=1.0):
            print(a)
            print(b)
            print(c)
            print(d)
        assert(d >= 0 and d <=1.0) # to make sure it is in range
        return 1 - d
    if method == '1-d/max': 
        #return max(0.0, 1.0 - np.linalg.norm(np.array(a) - np.array(b)) / dist_max)
        return 1 - norm_euclidean_distance(a,b,dims,dist_max) # do this instead of just d for internal assertion purposes
    d = np.linalg.norm(np.array(a) - np.array(b))
    if method == 'exp(-d)':
        return np.exp(-d)
    else: #if method == '1/(1+d)':
        #print('---',d,1.0 / (1.0 + d))
        return 1.0 / (1.0 + d)


# 'cos': The worst - this illustrates that cosine similarity is not usable completely
# it does not distinguish (10,10) and (1,1)
assert(float(round(norm_similarity((10,10),(10,10),'cos'),2))==1.0)
assert(float(round(norm_similarity((10,10),(10,9 ),'cos'),2))==1.0)
assert(float(round(norm_similarity((10,10),(10,5 ),'cos'),2))==0.95)
assert(float(round(norm_similarity((10,10),( 9,9 ),'cos'),2))==1.0) # problem!
assert(float(round(norm_similarity((10,10),(10,0 ),'cos'),2))==0.71)
#assert(np.isnan(float(round(norm_similarity((10,10),( 0,0 ),'cos'),2)))) # problem (commented out to suppress warning)!
assert(float(round(norm_similarity((10,),(1,),'cos'),2))==1.0) # problem!
assert(float(round(norm_similarity((10,10),(1,1),'cos'),2)==1.0)) # problem!
assert(float(round(norm_similarity((10,10,0,0,0),(1,1,0,0,0),'cos'),2))==1.0) # problem!
assert(float(round(norm_similarity((10,10,5,5,5),(1,1,5,5,5),'cos'),2))==0.65)

# '1/(1+d)': Not than good?
assert(float(round(norm_similarity((10,10),(10,10)),2))==1.0)
assert(float(round(norm_similarity((10,10),(10,9)),2))==0.5) # not good...
assert(float(round(norm_similarity((10,10),(10,5)),2))==0.17)
assert(float(round(norm_similarity((10,10),( 9,9 )),2))==0.41)
assert(float(round(norm_similarity((10,10),(10,0 )),2))==0.09)
assert(float(round(norm_similarity((10,10),( 0,0 )),2))==0.07)
assert(float(round(norm_similarity((10,),(1,)),2))==0.1)
assert(float(round(norm_similarity((10,10),(1,1)),2))==0.07)
assert(float(round(norm_similarity((10,10,0,0,0),(1,1,0,0,0)),2))==0.07)
assert(float(round(norm_similarity((10,10,5,5,5),(1,1,5,5,5)),2))==0.07)
assert(round(norm_similarity((10,10,5,5,5),(10,10,5,5,5)),2)==1.0)
assert(round(norm_similarity((10,10,5,5,5),(10,10,5,5,4)),2)==0.5) # not good...

# '1-d/max': The best!?
assert(float(round(norm_similarity((10,10),(10,10),'1-d/max',(10,10)),2))==1.0)
assert(float(round(norm_similarity((10,10),(10,9),'1-d/max',(10,10)),2))==0.93)
assert(float(round(norm_similarity((10,10),(10,5),'1-d/max',(10,10)),2))==0.65)
assert(round(norm_similarity((10,10),( 9,9 ),'1-d/max',(10,10)),2)==0.9)
assert(round(norm_similarity((10,10),(10,0 ),'1-d/max',(10,10)),2)==0.29)
assert(round(norm_similarity((10,10),( 0,0 ),'1-d/max',(10,10)),2)==0.0)
assert(round(norm_similarity((10,),(1,),'1-d/max',(10,)),2)==0.1)
assert(float(round(norm_similarity((10,10),(1,1),'1-d/max',(10,10)),2))==0.1)
assert(round(norm_similarity((10,10,0,0,0),(1,1,0,0,0),'1-d/max',   (10,10,10,10,10)),2)==0.43)
assert(round(norm_similarity((10,10,10,10,10),(1,1,1,1,1),'1-d/max',(10,10,10,10,10)),2)==0.1)
assert(round(norm_similarity((10,10,5,5,5),(10,10,5,5,5),'1-d/max', (10,10,10,10,10)),2)==1.0)
assert(round(norm_similarity((10,10,5,5,5),(1,1,5,5,5),'1-d/max',   (10,10,10,10,10)),2)==0.43)
assert(round(norm_similarity((10,10,5,5,5),(10,10,5,5,5),'1-d/max', (10,10,10,10,10)),2)==1.0)
assert(round(norm_similarity((10,10,5,5,5),(10,10,5,5,4),'1-d/max', (10,10,10,10,10)),2)==0.96) # good!
assert(round(norm_similarity([10,10,10],[10,10,10],'1-d/max',[10,10,10]),2)==1.0)
assert(round(norm_similarity([10,10,INT_NONE],[10,10,10],'1-d/max',[10,10,10]),2)==0.42)
assert(round(norm_similarity([10,INT_NONE,INT_NONE],[10,10,10],'1-d/max',[10,10,10]),2)==0.18)


def dimensions_init(N):
    return tuple([None] * 2 for _ in range(N))

def measure_dimensions(min_max,state):
    for i, var in enumerate(state):
        mm = min_max[i]
        if mm[0] is None or mm[0] > var: # min
            mm[0] = var
        if mm[1] is None or mm[1] < var: # max
            mm[1] = var

_space_min_max = dimensions_init(2)
measure_dimensions(_space_min_max,(10,10))
measure_dimensions(_space_min_max,(1,100))
assert(str(_space_min_max)=="([1, 10], [10, 100])")

# Model

def model_set_context_size(model,context_size=1):
    if context_size > 1:
        if not 'contexts' in model: 
            model['contexts'] = {}
            for size in range(2,context_size+1):
                if not size in model['contexts']:
                    model['contexts'][size] = {}
    return model

def model_new(context_size=1):
    """
    games: games count
    steps: steps count
    states: maps states to (utility,count,transtions) triple
        transitions: maps states pair to (utility,count)
    contexts: maps sizes of contexts (state series of size from 2 and more) respective contexts
        contexts: states to (utility,count,transtions) triple
            transitions: maps ... TODO
    """
    model = {'steps':0, 'games':0, 'states':{}}
    model_set_context_size(model,context_size=context_size)
    return model

def model_read_file(model_name):
    model_path = model_name + '.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        return model_new()

def model_write_file(model_name, model):
    model_path = model_name + '.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

#TODO optional exponentional reward decay OR remove if not used? 
def model_add_states(model,states,global_feedback):
    """
    Add potentially emotionally valuable episode of N of states with some global feeddback.
    Not used.
    """
    model['steps'] += len(states)
    model_states = model['states']
    previous = None
    for state in states:
        if state in model_states:
            (utility, count, transitions) = model_states[state]
            model_states[state] = (utility + global_feedback, count + 1, transitions)
        else:
            model_states[state] = (global_feedback, 1, {})
        if not previous is None:
            (utility, count, transitions) = model_states[previous]
            if state in transitions:
                (transition_utility, transition_count) = transitions[state]
                transitions[state] = (transition_utility + global_feedback, transition_count + 1)
            else:
                transitions[state] = (global_feedback, 1)
            model_states[previous] = (utility, count, transitions)
        previous = state
    return model

assert(str(model_add_states(model_new(),[],0))=="{'steps': 0, 'games': 0, 'states': {}}")
assert(str(model_add_states(model_new(),[(0,0),(0,1)],0))=="{'steps': 2, 'games': 0, 'states': {(0, 0): (0, 1, {(0, 1): (0, 1)}), (0, 1): (0, 1, {})}}")
assert(str(model_add_states(model_new(),[(0,0),(0,1),(0,0),(0,1)],0))=="{'steps': 4, 'games': 0, 'states': {(0, 0): (0, 2, {(0, 1): (0, 2)}), (0, 1): (0, 2, {(0, 0): (0, 1)})}}")
assert(str(model_add_states(model_new(),[(0,0),(0,1),(0,0),(0,1)],1))=="{'steps': 4, 'games': 0, 'states': {(0, 0): (2, 2, {(0, 1): (2, 2)}), (0, 1): (2, 2, {(0, 0): (1, 1)})}}")

#TODO optional exponentional reward decay?
def model_add_state_transitions(model_states,previous,state,global_feedback):
    """
    Add potentially emotionally valuable single state and its transition to another state with some global feeddback.
    """
    if state in model_states:
        (utility, count, transitions) = model_states[state]
        model_states[state] = (utility + global_feedback, count + 1, transitions)
    else:
        model_states[state] = (global_feedback, 1, {})
    if not previous is None:
        (utility, count, transitions) = model_states[previous]
        if state in transitions:
            (transition_utility, transition_count) = transitions[state]
            transitions[state] = (transition_utility + global_feedback, transition_count + 1)
        else:
            transitions[state] = (global_feedback, 1)
        model_states[previous] = (utility, count, transitions)

def model_add_context_transitions(model_contexts,context,state,global_feedback):
    """
    Add potentially emotionally valuable series of states (context) and its transition to a single state with some global feeddback.
    """
    if context in model_contexts:
        (utility, count, transitions) = model_contexts[context]
        utility += global_feedback
        count += 1
    else:
        (utility, count, transitions) = (global_feedback, 1, {})
    if state in transitions:
        (transition_utility, transition_count) = transitions[state]
        transitions[state] = (transition_utility + global_feedback, transition_count + 1)
    else:
        transitions[state] = (global_feedback, 1)
    model_contexts[context] = (utility, count, transitions)

def model_add_states_contexts(model,states,global_feedback):
    """
    Add emotionally valuable episode of N of states with some global feeddback as transitions from state to state and
    from series of states (context) to single state. 
    """
    model['steps'] += len(states)
    model_states = model['states']
    model_contexts = model['contexts'] if 'contexts' in model else None
    model_contexts_count = len(model_contexts) if not model_contexts is None else 0
    previous = None
    for index, state in enumerate(states):
        model_add_state_transitions(model_states,previous,state,global_feedback)
        previous = state
        for context_index in range(model_contexts_count): # from 2 and up, [0,1,2,...]=>[2,3,4,...]
            context_size = context_index + 2
            if index + 1 > context_size: #TODO simplify
                model_context = model_contexts[context_size]
                context = sum(states[index-context_size:index],()) # concatentate series of states into continious state for sompler matching
                model_add_context_transitions(model_context,context,state,global_feedback)
    return model

assert(str(model_add_states_contexts(model_new(),[],0))=="{'steps': 0, 'games': 0, 'states': {}}")
assert(str(model_add_states_contexts(model_new(),[(0,0),(0,1)],0))=="{'steps': 2, 'games': 0, 'states': {(0, 0): (0, 1, {(0, 1): (0, 1)}), (0, 1): (0, 1, {})}}")
assert(str(model_add_states_contexts(model_new(),[(0,0),(0,1),(0,0),(0,1)],0))=="{'steps': 4, 'games': 0, 'states': {(0, 0): (0, 2, {(0, 1): (0, 2)}), (0, 1): (0, 2, {(0, 0): (0, 1)})}}")
assert(str(model_add_states_contexts(model_new(),[(0,0),(0,1),(0,0),(0,1)],1))=="{'steps': 4, 'games': 0, 'states': {(0, 0): (2, 2, {(0, 1): (2, 2)}), (0, 1): (2, 2, {(0, 0): (1, 1)})}}")
assert(str(model_new(context_size=2))=="{'steps': 0, 'games': 0, 'states': {}, 'contexts': {2: {}}}")
assert(str(model_add_states_contexts(model_new(context_size=2),[(1,1),(2,2),(3,3)],0))=="{'steps': 3, 'games': 0, 'states': {(1, 1): (0, 1, {(2, 2): (0, 1)}), (2, 2): (0, 1, {(3, 3): (0, 1)}), (3, 3): (0, 1, {})}, 'contexts': {2: {(1, 1, 2, 2): (0, 1, {(3, 3): (0, 1)})}}}")
assert(str(model_add_states_contexts(model_new(context_size=2),[(1,1),(2,2),(3,3),(4,4)],1)['contexts'])=="{2: {(1, 1, 2, 2): (1, 1, {(3, 3): (1, 1)}), (2, 2, 3, 3): (1, 1, {(4, 4): (1, 1)})}}")
assert(str(model_add_states_contexts(model_new(context_size=2),[(1,1),(2,2),(1,1),(2,2),(3,3)],1)['contexts'])=="{2: {(1, 1, 2, 2): (2, 2, {(1, 1): (1, 1), (3, 3): (1, 1)}), (2, 2, 1, 1): (1, 1, {(2, 2): (1, 1)})}}")
assert(str(model_add_states_contexts(model_new(context_size=3),[(1,1),(2,2),(1,1),(2,2),(3,3)],1)['contexts'])=="{2: {(1, 1, 2, 2): (2, 2, {(1, 1): (1, 1), (3, 3): (1, 1)}), (2, 2, 1, 1): (1, 1, {(2, 2): (1, 1)})}, 3: {(1, 1, 2, 2, 1, 1): (1, 1, {(2, 2): (1, 1)}), (2, 2, 1, 1, 2, 2): (1, 1, {(3, 3): (1, 1)})}}")

def find_similar(states,state,state_count_threshold,state_similarity_threshold,sim_method,dims,max_dist):
    max_sim = 0
    bests = []
    for s, utility_count in states.items():
        if utility_count[1] < state_count_threshold: # disregard rare evidence
            continue
        sim = norm_similarity(s,state,sim_method,dims,max_dist)
        if sim < state_similarity_threshold:
            continue
        if max_sim < sim:
            max_sim = sim
            bests.clear()
            bests.append(s)
        elif max_sim == sim:
            bests.append(s)
    best = bests[0] if len(bests) == 1 else random.choice(bests) if len(bests) > 1 else None
    return states[best] if not best is None else None


def find_useful(transitions,transition_utility_thereshold,transition_count_threshold):
    max_utility = -1000000000
    max_count = 0
    bests = []
    for s, utility_count in transitions.items():
        utility, count = utility_count
        if utility < transition_utility_thereshold: # disregard low utility
            continue
        if count < transition_count_threshold: # disregard rare evidence
            continue
        if max_utility < utility:
            max_utility = utility
            max_count = count
            bests.clear()
            bests.append(s)
        elif max_utility == utility:
            bests.append(s)
    best = bests[0] if len(bests) == 1 else random.choice(bests) if len(bests) > 1 else None
    if not best is None:
        #print('found',max_utility,max_count,len(transitions),best[0] if not best is None else '-')
        pass
    return best


def find_useful_action(actions,transitions,transition_utility_thereshold,transition_count_threshold,debug = False):
    actions_uc = {a:0 for a,k in enumerate(actions)} # TODO: optimize to arrray from map!?
    for s, utility_count in transitions.items():
        utility, count = utility_count
        if (not transition_utility_thereshold is None) and utility < transition_utility_thereshold: # disregard low utility
            continue
        if count < transition_count_threshold: # disregard rare evidence
            continue
        actions_uc[s[0]] += utility * count
        #actions_uc[s[0]] += utility
    max_uc = None
    acts = []
    for a in actions_uc:
        uc = actions_uc[a]
        if max_uc is None or max_uc < uc:
            max_uc = uc
            acts.clear()
            acts.append(a)
        elif max_uc == uc:
            acts.append(a)
    act = acts[0] if len(acts) == 1 else random.choice(acts)
    #print(str(actions_uc),str({a:round(actions_uc[a]) for a in acts}),act)
    return act


# TODO make abstract
class GymPlayer:
    def __init__(self,debug):
        self.debug = debug

    def process_state(self, observation, reward, previous_action):
        pass

    def debugging(self):
        if self.debug: #or os.path.exists('debug'): # avoid OS hitting HDD!?
            return True
        return False 

if __name__ == "__main__":
    print("Ok")