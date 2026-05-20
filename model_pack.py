import ale_py
import gymnasium as gym
import numpy as np
import datetime as dt
import argparse
import subprocess

from basic import *


def dictcount(dic,arg,cnt=1):
    if type(arg) == list:
        #print(arg)
        #print(dic)
        for i in arg:
            dictcount(dic,i,cnt)
    elif arg in dic:
        dic[arg] = dic[arg] + cnt
    else:
        dic[arg] = cnt

def roundup(x,den):
    return int(round(x / den)) * den

parser = argparse.ArgumentParser(description='Open AI Gym Evaluator')
parser.add_argument('-i','--input', type=str, default=None, help='Input model')
parser.add_argument('-o','--output', type=str, default="model", help='Output model')
parser.add_argument('-cs','--context_size', type=int, default=1, help='Context size')
parser.add_argument('-sc','--state_count', type=int, default=2, help='State count threshold')
parser.add_argument('-ss','--state_similarity', type=float, default=0.9, help='State similarity threshold')
parser.add_argument('-tu','--transition_utility', type=int, default=None, help='Transition utility thereshold')
parser.add_argument('-tc','--transition_count', type=int, default=1, help='Transition count threshold')
parser.add_argument('-su','--state_utility', type=int, default=None, help='State utility thereshold')

args = parser.parse_args()

model = model_new() if args.input is None else model_read_file(args.input)

print(f"model=\"{args.input}\"; states={len(model['states'])}; games={model['games']}; steps={model['steps']}")
print(1,len(model['states']))


def print_model(model,rounding=1000):
    if 'contexts' in model:
        contexts = model['contexts']
        for cs in contexts:
            contexts_cs = contexts[cs]
            scounts = {}
            ucounts = {}
            tcounts = {}
            sum = 0
            for ct in contexts_cs:
                (utility,count,transitions) = contexts_cs[ct] 
                tcount = len(transitions)   
                sum += tcount 
                dictcount(ucounts,roundup(utility,rounding))
                dictcount(scounts,roundup(count,rounding))
                dictcount(tcounts,tcount)
            print(cs,len(contexts_cs),round(sum/len(contexts_cs),2))
            print("scounts=",scounts)
            print("ucounts=",ucounts)
            print("tcounts=",tcounts)

def pack_model(model,args):
    if 'contexts' in model:
        contexts = model['contexts']
        for cs in contexts:
            new = {}
            contexts_cs = contexts[cs]
            scounts = {}
            ucounts = {}
            sum = 0
            for ct in contexts_cs:
                (utility,count,transitions) = contexts_cs[ct]   
                sum += len(transitions)
                if not args.state_utility is None and utility < args.state_utility:
                    continue
                new[ct] = (utility,count,transitions)
            contexts[cs] = new

print_model(model)
pack_model(model,args)
print_model(model)

if not (args.output is None or args.output == args.input):
    model_write_file(args.output,model)

