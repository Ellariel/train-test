import os, argparse, pickle
#import networkx as nx
#time, sys, gym, random, , 
import numpy as np
#import pandas as pd
#from gym import spaces
from tqdm import tqdm
#from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.monitor import Monitor

parser = argparse.ArgumentParser()
parser.add_argument('--approach', default='PPO', type=str)
parser.add_argument('--n_envs', default=16, type=int)
parser.add_argument('--env', default='env', type=str)

parser.add_argument('--subgraph', default=50, type=int)
parser.add_argument('--idx', default=0, type=int)
parser.add_argument('--subset', default='randomized', type=str)

args = parser.parse_args()

n_envs = args.n_envs
approach = args.approach
subgraph = args.subgraph
idx = args.idx
subset = args.subset

if args.env == 'env':
    version='env'
    from env import LNEnv

def max_neighbors(G):
    def neighbors_count(G, id):
        return len(list(G.neighbors(id)))
    max_neighbors = 0
    for id in G.nodes:
      max_neighbors = max(max_neighbors, neighbors_count(G, id))
    return max_neighbors

def test_path(u, v, amount=100):
    E_.subset = [(u, v, amount)]
    obs = E_.reset()
    action, _states = model.predict(obs, deterministic=True)
    path = E_.predict_path(action)
    if E_.check_path():
        return path
    
    #if v in path:
    #    return path
    #obs, reward, done, info = E_.step(action)
    #return E_.predict_path(action)
    #if E_.check_path():
    #       return E_.get_path()  

base_dir = './'
snapshots_dir = os.path.join(base_dir, 'snapshots')
weights_dir = os.path.join(base_dir, 'weights')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

with open(os.path.join(snapshots_dir, f'sampled_graph.pickle'), 'rb') as f:
    samples = pickle.load(f)
    print(f"available subgraphs: {', '.join([str(i) for i in samples.keys()])}")
    sample = samples[subgraph]
    print(f"{subgraph}:")

G = sample[idx]['subgraph']
T = sample[idx]['subgraph_transactions'][subset]

train_size = int(len(T) * 0.6)
test_size = int(len(T) * 0.2)
train_set = T[:train_size]
test_set = T[train_size:train_size+test_size]
valid_set = T[train_size+test_size:]

print(f'subgraph, n: {len(G.nodes)}, e: {len(G.edges)}, max neighbors: {max_neighbors(G)}, sample idx: {idx}')
print(f'transations count: {len(T)}, train_set: {len(train_set)}, test_set: {len(test_set)}, valid_set: {len(valid_set)}')

file_mask = f'{approach}-{version}-{n_envs}-{subset}-{subgraph}-{idx}'

E_ = LNEnv(G, [], train=False)

#e = LNEnv(G, T, train=True)
#check_env(e)

f = os.path.join(weights_dir, f'{file_mask}.sav')

if approach == 'PPO':
        model_class = PPO
else:
        model_class = None
        print(f'{approach} - not implemented!')
        raise ValueError

if os.path.exists(f) and model_class:
        model = model_class.load(f, E_, force_reset=False)
        print(f'model is loaded {approach}: {f}')
else:
        print(f'did not find {approach}: {f}')
        model = model_class("MlpPolicy", E_) 

train_score = 0 
total_pathlen = []
for tx in tqdm(train_set):
            r = test_path(tx[0], tx[1], tx[2])
            if r:
                train_score += 1
                total_pathlen += [len(r)]
train_score = train_score / len(train_set)
print(f'pathlen min: {min(total_pathlen)} average: {np.mean(total_pathlen)} max: {max(total_pathlen)}')

test_score = 0 
total_pathlen = []
for tx in tqdm(test_set):
            r = test_path(tx[0], tx[1], tx[2])
            if r:
                test_score += 1
                total_pathlen += [len(r)]
test_score = test_score / len(test_set)
print(f'pathlen min: {min(total_pathlen)} average: {np.mean(total_pathlen)} max: {max(total_pathlen)}')
        
valid_score = 0 
total_pathlen = []
for tx in tqdm(valid_set):
            r = test_path(tx[0], tx[1], tx[2])
            if r:
                valid_score += 1
                total_pathlen += [len(r)]
valid_score = valid_score / len(valid_set)
print(f'pathlen min: {min(total_pathlen)} average: {np.mean(total_pathlen)} max: {max(total_pathlen)}')

        
print(f'''v: {version}
          train score: {train_score:.3f},
          test score: {test_score:.3f},
          validation score: {valid_score:.3f}
          ''')


