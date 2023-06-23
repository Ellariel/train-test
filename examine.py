import networkx as nx
import numpy as np
import pandas as pd
import os, time, pickle, glob, random, argparse
from tqdm import tqdm
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
# https://github.com/mlco2/codecarbon#start-to-estimate-your-impact-
# https://arxiv.org/pdf/1911.08354.pdf

from proto.eclair import EclairRouting
from proto.clightning import CLightningRouting
from proto.lnd import LNDRouting

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

def to_bidirected(G):
    G = nx.DiGraph(G)
    for u, v in G.edges:
        if not G.has_edge(v, u):
            G.add_edge(v, u)
            attrib = {(v, u) : G.edges[u, v]}
            nx.set_edge_attributes(G, attrib)
    return G
    
class RLRouting():
    def __init__(self, G, weights_dir, approach='PPO', version='env', n_envs=16, subset='randomized', subgraph=50, idx=0) -> None:
        self.version = version
        self.approach = approach
        self.n_envs = n_envs
        self.subset = subset
        self.subgraph = subgraph
        self.idx = idx
        self.g = G
        self.env = LNEnv(self.g, [], train=False)
        self.model = None
        
        self.file_mask = f'{approach}-{version}-{n_envs}-{subset}-{subgraph}-{idx}'
        f = os.path.join(weights_dir, f'{self.file_mask}.sav')
        if self.approach == 'PPO':
                self.model_class = PPO
        else:
                self.model_class = None
                print(f'{self.approach} - not implemented!')
                raise ValueError

        if os.path.exists(f) and self.model_class:
                self.model = self.model_class.load(f, self.env, force_reset=False)
                print(f'model is loaded {self.approach}: {f}')
        else:
                print(f'did not find {self.approach}: {f}')
                self.model = self.model_class("MlpPolicy", self.env)         
           
    def name(self):
        return f"RL-agent({self.approach})"           
        
    def get_delay(self, u, v):
        e = self.g.edges[u, v]
        delay = 0
        if 'delay' in e:
            delay += e["delay"]
        return delay

    def get_amount(self, u, v, amount):
        e = self.g.edges[u, v]
        fee = 0
        if 'fee_base_sat' in e:
            fee = e['fee_base_sat'] + amount * e['fee_rate_sat']
        return fee

    def get_total_delay(self, path):
        total_delay = 0
        for i in range(len(path)-1):
            if [path[i], path[i + 1]] in self.g.edges:
                total_delay += self.get_delay(path[i], path[i + 1])
        return total_delay
        
    def get_total_amount(self, path, amount):
        total_amount = amount
        for i in range(1, len(path)-1):
            if [path[i], path[i + 1]] in self.g.edges:
                total_amount += self.get_amount(path[i], path[i + 1], total_amount)
        return total_amount

    def routePath(self, G, u, v, amount):
        r = {'path' : None, 'ok' : False}
        if (not self.env) or (not self.model):
            print('Model is not initialized!')
            raise ValueError
        
        start_time = time.time()
        self.env.subset = [(u, v, amount)]       
        obs = self.env.reset()
        action, _states = self.model.predict(obs, deterministic=True)
        obs, reward, done, info = self.env.step(action)
        if self.env.check_path():
                r["path"] = self.env.get_path()
                r["runtime"] = time.time() - start_time
                r["dist"] = len(r["path"])
                r["ok"] = r["dist"] > 0
                r["u"] = u
                r["v"] = v
                if r["ok"]:
                        r["delay"] = self.get_total_delay(r["path"])
                        r["amount"] = self.get_total_amount(r["path"], amount)
                        r["feeratio"] = r["amount"] / amount
                        r["feerate"] = r["amount"] / amount - 1
                        r["amount"] = amount
        return r  
    
def track_emissions(G, T, routingObj, alg):
    results = []
    with OfflineEmissionsTracker(country_iso_code="CAN", measure_power_secs=1, tracking_mode='process') as tracker:
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
    
    algorithms = {'RLA': RLRouting(G, weights_dir, approach=approach, 
                                version=version, n_envs=n_envs, 
                                subset=subset, subgraph=subgraph, 
                                idx=i),
                 'LND': LNDRouting(),
                 'CLN': CLightningRouting(random.uniform(-1,1)),
                 'ECL': EclairRouting(),
                 }
    results = {}
    emissions_idx = []
    for algorithm, _routingObj in tqdm(algorithms.items(), leave=False):
        t = T.copy()
        g = G.copy() if algorithm == 'RLA' else to_bidirected(G)
        
        results[f"{algorithm}-{subgraph}-{i}"] = track_emissions(g, t, _routingObj, _routingObj.name())
        emissions_idx.append(f"{algorithm}-{subgraph}-{i}")    
    
        if os.path.exists(os.path.join(results_dir, 'emissions.csv')): 
            e = pd.read_csv(os.path.join(results_dir, 'emissions.csv'))
            e = pd.concat([e, pd.Series(emissions_idx, name='emissions_idx')], axis=1)
            e = e[['timestamp', 'duration', 'emissions', 
                'emissions_rate', 'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy',
                'gpu_energy', 'ram_energy', 'energy_consumed', 'emissions_idx']]
            results[f"{algorithm}-{subgraph}-{i}-emissions"] = e.to_dict()
            os.remove(os.path.join(results_dir, 'emissions.csv'))

    with open(os.path.join(results_dir, f"results-{subset}-{subgraph}-{i}.pickle"), 'wb') as f:
            pickle.dump(results, f)

print('done')
print(f"git add -f results-*.pickle && git commit -m results && git push")