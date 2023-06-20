import gymnasium as gym, ray
from ray import air, tune
from gymnasium.wrappers import EnvCompatibility
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

import os, time, sys, random, pickle, argparse
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--approach', default='RPPO', type=str)
parser.add_argument('--n_envs', default=1, type=int)
parser.add_argument('--env', default='renv', type=str)

parser.add_argument('--subgraph', default=50, type=int)
parser.add_argument('--idx', default=0, type=int)
parser.add_argument('--subset', default='randomized', type=str)

parser.add_argument('--attempts', default=1, type=int)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--timesteps', default=10, type=int)

args = parser.parse_args()

idx = args.idx
n_envs = args.n_envs
timesteps = args.timesteps
approach = args.approach
epochs = args.epochs
attempts = args.attempts
subgraph = args.subgraph
subset = args.subset

if args.env == 'renv':
    version = 'renv'
    from renv import LNEnv
    
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
    action, _states, _ = model.compute_single_action(obs, explore=False)#.predict(obs, deterministic=True)
    obs, reward, done, info = E_.step(action)
    if E_.check_path():
           return E_.get_path()  

base_dir = './'
snapshots_dir = os.path.join(base_dir, 'snapshots')
weights_dir = os.path.join(base_dir, 'weights')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

#samples = nx.read_gpickle(os.path.join(snapshots_dir, f'graph-sample-{subgraph}.pickle'))
with open(os.path.join(snapshots_dir, f'sampled_graph.pickle'), 'rb') as f:
    samples = pickle.load(f)
    print(f"available subgraphs: {', '.join([str(i) for i in samples.keys()])}")

sample = samples[subgraph]
print(f"{subgraph}:")
for id, i in enumerate(sample):
        g = i['subgraph']
        print(f'idx: {id}, n: {len(g.nodes)}, e: {len(g.edges)}, max_neighbors: {max_neighbors(g)}')
print('')

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

ray.init()
def env_creator(env_config):
    return EnvCompatibility(LNEnv(G = env_config['G'],
                 transactions = env_config['T'],
                 train = env_config['train']))  # return an env instance
register_env("LNEnv", env_creator)
   
    
for a in range(attempts):
    print(f"approach: {approach}, env: {version}, n_envs: {n_envs}, subset: {subset}, subgraph: {subgraph}, sample idx: {idx}")
    print(f"train: {file_mask}")

    E_ = LNEnv(G, [], train=False)
    #check_env(E_)    
    #E = make_vec_env(lambda: LNEnv(G, train_set), n_envs=n_envs)


    lf = os.path.join(results_dir, f'{file_mask}.log')
    log = pd.read_csv(lf, sep=';', compression='zip') if os.path.exists(lf) else None
    f = os.path.join(weights_dir, f'{file_mask}.sav')

    if approach == 'RPPO':
        model_class = ppo.PPO
    else:
        model_class = None
        print(f'{approach} - not implemented!')
        raise ValueError

    if os.path.exists(f) and model_class:
        #model = model_class.load(f, E, force_reset=False, verbose=0, learning_rate=learning_rate)
        print(f'model is loaded {approach}: {f}')
    else:
        print(f'did not find {approach}: {f}')
        #model = model_class("MlpPolicy", E, verbose=0, learning_rate=learning_rate) 
        '''
        model = model_class(env='LNEnv', config={"env_config": {
                                                    'G': G,
                                                    'T': train_set,
                                                    'train': True,
    
                                                    'num_gpus': 1,
                                                    'num_workers': 4,
                                                    
                                                    'vf_clip_param': 500.0,
                                                    'lambda': 0.1,
                                                    'gamma': 0.99,
                                                    'lr': 0.000001,
            },  # config to pass to env class
        '''
        model = model_class(config = AlgorithmConfig()
                            .environment(env = "LNEnv", 
                                         clip_rewards = 500.0,
                                         env_config = {'G': G,
                                                       'T': train_set,
                                                       'train': True,
                                                      },
                                         )
                            .training(gamma = 0.9, 
                                      lr = 0.01,
                                      )
                            .resources(num_gpus = 0)
                            .rollouts(num_rollout_workers = 4)
                            .build())
                            #.callbacks(MemoryTrackingCallbacks)
# A config object can be used to construct the respective Trainer.

                                                 
                                                 
#        })

    for epoch in range(1, epochs + 1):
        #model.learn(total_timesteps=timesteps, progress_bar=True)
        for step in tqdm(range(int(timesteps))):
            result = model.train()
        print(pretty_print(result))
        
        train_score = 0
        train_total_pathlen = []
        for tx in tqdm(train_set):
            r = test_path(tx[0], tx[1], tx[2])
            if r:
                train_score += 1
                train_total_pathlen += [len(r)]
        train_score = train_score / len(train_set)

        test_score = 0 
        test_total_pathlen = []
        for tx in tqdm(test_set):
            r = test_path(tx[0], tx[1], tx[2])
            if r:
                test_score += 1
                test_total_pathlen += [len(r)]
        test_score = test_score / len(test_set)

        reward = E.env_method('get_reward')
        mean_reward = np.mean(reward, axis=1)
        max_mean_reward = np.max(mean_reward)
        
        print(f'n_envs: {n_envs}, epoch: {epoch}/{epochs}, attempt: {a}/{attempts}')        
        print(f'test score: {test_score}, pathlen min: {np.min(test_total_pathlen)} average: {np.mean(test_total_pathlen):.1f} max: {np.max(test_total_pathlen)}')
        print(f'train score: {train_score}, pathlen min: {np.min(train_total_pathlen)} average: {np.mean(train_total_pathlen):.1f} max: {np.max(train_total_pathlen)}')
        print(f"max mean reward: {max_mean_reward:.3f}~{mean_reward}")
        model.save(f)

        if max(train_score, test_score) > 0.7:
            model.save(f + f'-{train_score:.3f}-{test_score:.3f}')
            print('saved:', f + f'-{train_score:.3f}-{test_score:.3f}')

        log = pd.concat([log, pd.DataFrame.from_dict({'time' : time.time(),
                                                'approach' : approach,
                                                'version' : version,
                                                'n_envs': n_envs,
                                                'subset' : subset,
                                                'subgraph' : subgraph,
                                                'idx' : idx,
                                                'max_mean_reward' : max_mean_reward,
                                                'mean_reward' : mean_reward,
                                                'test_score' : test_score,
                                                'train_score' : train_score,
                                                'epoch' : epoch,
                                                'epochs' : epochs,
                                                'attempt' : a,
                                                'total_timesteps' : timesteps,
                                                'filename' : f,
                                                'n': len(G.nodes),
                                                'e': len(G.edges),
                                                'max_neighbors':max_neighbors(G),
                                                'min_pathlen': min(test_total_pathlen + train_total_pathlen),
                                                'max_pathlen': max(test_total_pathlen + train_total_pathlen),
                                                'avg_pathlen': np.mean(test_total_pathlen + train_total_pathlen),
                                                }, orient='index').T], ignore_index=True)
        log.to_csv(lf, sep=';', index=False, compression='zip')