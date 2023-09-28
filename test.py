import os, argparse, pickle, time, glob, sys
import numpy as np
from tqdm import tqdm
#from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC

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

def test_path(u, v, amount=100, max_path_length=100):
    E_.subset = [(u, v, amount)]
    obs = E_.reset()
    action, _states = model.predict(obs, deterministic=True)
    path = E_.predict_path(action, max_path_length=max_path_length)
    if E_.check_path():
        return path

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

weights_file = os.path.join(weights_dir, f'{file_mask}.sav')
weights_file_list = glob.glob(weights_file + '-*')
if len(weights_file_list):
    weights_file_list = sorted(weights_file_list, key=lambda x: float(''.join([i for i in x.split('-')[-1] if i.isdigit() or i == '.'])))
    if os.path.exists(weights_file_list[-1]):
        weights_file = weights_file_list[-1]

print(weights_file)

if approach == 'PPO':
        model_class = PPO
else:
        model_class = None
        print(f'{approach} - not implemented!')
        raise ValueError

if os.path.exists(weights_file) and model_class:
        model = model_class.load(weights_file, E_, force_reset=False)
        print(f'model is loaded {approach}: {weights_file}')
else:
        print(f'did not find {approach}: {weights_file}')
        model = model_class("MlpPolicy", E_) 

def _test(_set):
    _score = 0 
    total_pathlen = []
    run_time = []
    for tx in tqdm(_set):
                start_time = time.time()
                r = test_path(tx[0], tx[1], tx[2])
                if r:
                    _score += 1
                    total_pathlen += [len(r)]
                run_time += [time.time() - start_time]
    _score = _score / len(_set)
    print(f'set size: {len(_set)}')
    print(f'pathlen min: {min(total_pathlen)} average: {np.mean(total_pathlen):.2f} max: {max(total_pathlen)}')
    print(f'average runtime: {np.mean(run_time)}')
    return _score

print(f'''v: {version}
          train score: {_test(train_set):.3f},
          test score: {_test(test_set):.3f},
          validation score: {_test(valid_set):.3f}
          ''')


