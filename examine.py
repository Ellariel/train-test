import numpy as np
import pandas as pd
import networkx as nx
import os, time, pickle, glob, random, argparse
from tqdm import tqdm
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from codecarbon import OfflineEmissionsTracker
# https://github.com/mlco2/codecarbon#start-to-estimate-your-impact-
# https://arxiv.org/pdf/1911.08354.pdf

import warnings
warnings.filterwarnings('ignore')

from proto.eclair import EclairRouting
from proto.clightning import CLightningRouting
from proto.lnd import LNDRouting

parser = argparse.ArgumentParser()
parser.add_argument('--approach', default='PPO', type=str)
parser.add_argument('--n_envs', default=16, type=int)
parser.add_argument('--env', default='env', type=str)

parser.add_argument('--subgraph', default=50, type=int)
parser.add_argument('--idx', default=3, type=int)
parser.add_argument('--subset', default='randomized', type=str)
args = parser.parse_args()

idx = args.idx
subset = args.subset
n_envs = args.n_envs
approach = args.approach
subgraph = args.subgraph

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

def reverse(G):
    G = nx.DiGraph(G)
    attrib = {}
    for u, v in G.edges:
        if not G.has_edge(v, u):
            G.add_edge(v, u)
            attrib.update({(v, u) : G.edges[u, v]})
            G.remove_edge(u, v)
    nx.set_edge_attributes(G, attrib)
    return G
    
class RLRouting():
    def __init__(self, G, weights, approach='PPO') -> None:
        self.weights = weights
        self.approach = approach
        self.g = G
        self.env = LNEnv(self.g, [], train=False)
        self.model = None

        if self.approach == 'PPO':
                self.model_class = PPO
        else:
                self.model_class = None
                print(f'{self.approach} - not implemented!')
                raise ValueError

        if os.path.exists(self.weights) and self.model_class:
                self.model = self.model_class.load(self.weights, self.env, force_reset=False)
                print(f'model is loaded {self.approach}: {self.weights}')
        else:
                print(f'did not find {self.approach}: {self.weights}')
                self.model = self.model_class("MlpPolicy", self.env)         
           
    def name(self):
        return f"RL-agent({self.approach})"           
           
    def get_total(self, path, amount):
        total_delay = 0
        total_amount = amount
        for i in range(0, len(path)-1):
            if [path[i], path[i + 1]] in self.g.edges:
                e = self.g.edges[path[i], path[i + 1]]
                if 'delay' in e:
                    total_delay += e["delay"]
                if 'fee_base_sat' in e and i > 0:
                    total_amount += e['fee_base_sat'] + total_amount * e['fee_rate_sat']
        return total_delay, total_amount

    def routePath(self, G, u, v, amount):
        r = {'path' : None, 'ok' : False}
        if (not self.env) or (not self.model):
            print('Model is not initialized!')
            raise ValueError
        
        start_time = time.time()
        self.env.subset = [(u, v, amount)]       
        obs = self.env.reset()
        action, _states = self.model.predict(obs, deterministic=True)
        path = self.env.predict_path(action)
        if v in path:
                r["path"] = path
                r["runtime"] = time.time() - start_time
                r["dist"] = len(r["path"])
                r["ok"] = r["dist"] > 0
                r["u"] = u
                r["v"] = v
                if r["ok"]:
                        r["delay"], r["amount"] = self.get_total(r["path"], amount)
                        r["feeratio"] = r["amount"] / amount
                        r["feerate"] = r["amount"] / amount - 1
                        r["amount"] = amount
        return r  
    
def track_emissions(G, T, routingObj, alg):
    results = []
    with OfflineEmissionsTracker(country_iso_code="CAN", 
                                 measure_power_secs=0.5, 
                                 tracking_mode='process', 
                                 output_file=os.path.join(results_dir, 'emissions.csv')) as tracker:
        for t in tqdm(T, leave=False, desc=alg):
            results.append(routingObj.routePath(G, t[0], t[1], t[2]))
    return results

base_dir = './'
snapshots_dir = os.path.join(base_dir, 'snapshots')
weights_dir = os.path.join(base_dir, 'weights')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(snapshots_dir, f'sampled_graph.pickle'), 'rb') as f:
    samples = pickle.load(f)
    print(f"available subgraphs: {', '.join([str(i) for i in samples.keys()])}")
    sample = samples[subgraph]
    print(f"{subgraph}:")

for i in tqdm(range(idx+1), total=idx+1, leave=True):
    if os.path.exists(os.path.join(results_dir, 'emissions.csv')): 
        print('...remove old emissions.csv')
        os.remove(os.path.join(results_dir, 'emissions.csv'))
    
    G = sample[i]['subgraph']
    T = sample[i]['subgraph_transactions'][subset]
    
    random.seed(48)
    np.random.seed(48)    

    file_mask = f'{approach}-{version}-{n_envs}-{subset}-{subgraph}-{i}' 

    weights_file = os.path.join(weights_dir, f'{file_mask}.sav')
    weights_file_list = glob.glob(weights_file + '-*')
    if len(weights_file_list):
        weights_file_list = sorted(weights_file_list, key=lambda x: float(''.join([i for i in x.split('-')[-1] if i.isdigit() or i == '.'])))
        if os.path.exists(weights_file_list[-1]):
            weights_file = weights_file_list[-1]    
    
    if not os.path.exists(weights_file):
        continue
    
    algorithms = {'RLA': RLRouting(G, weights=weights_file),
                  'LND': LNDRouting(),
                  'CLN': CLightningRouting(random.uniform(-1,1)),
                  'ECL': EclairRouting(),
                 }
    results = {}

    for algorithm, _routingObj in tqdm(algorithms.items(), leave=False):
        t = T.copy()
        g = G.copy() if algorithm == 'RLA' else reverse(G)
        
        results[f"{algorithm}-{subgraph}-{i}"] = track_emissions(g, t, _routingObj, _routingObj.name())
        emissions_idx = [f"{algorithm}-{subgraph}-{i}"] 
    
        if os.path.exists(os.path.join(results_dir, 'emissions.csv')): 
            e = pd.read_csv(os.path.join(results_dir, 'emissions.csv'))
            e = pd.concat([e, pd.Series(emissions_idx, name='emissions_idx')], axis=1)
            e = e[['emissions_idx', 'timestamp', 'duration', 'emissions']]
            results[f"{algorithm}-{subgraph}-{i}-emissions"] = e.to_dict()
            os.remove(os.path.join(results_dir, 'emissions.csv'))

    with open(os.path.join(results_dir, f"{file_mask}.pickle"), 'wb') as f:
            pickle.dump(results, f)

print('done')
print(f"git add -f results/*.pickle && git commit -m results && git push")