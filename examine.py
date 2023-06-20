import networkx as nx
import numpy as np
import pandas as pd
import os, sys, time, pickle, glob, random, argparse
from tqdm import tqdm
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
# https://github.com/mlco2/codecarbon#start-to-estimate-your-impact-
# https://arxiv.org/pdf/1911.08354.pdf

from eclair import EclairRouting
from clightning import CLightningRouting
from lnd import LNDRouting

parser = argparse.ArgumentParser()
parser.add_argument('--agents', default=6, type=int)
parser.add_argument('--max_steps', default=10, type=int)
args = parser.parse_args()

agents = args.agents
max_steps = args.max_steps

routingObj = None
_env = None
def change_env(env, filename, snapshot):
    global routingObj, _env
    if env == '_env':
        from _env import LNEnv
        _env = LNEnv(snapshot, max_steps=max_steps, train=False)
        routingObj = RLRouting(snapshot, _env, max_steps=max_steps, load_from=filename)

def neighbors_count(G, id):
    return len(list(G.neighbors(id)))

def max_neighbors(G):
    max_neighbors = 0
    for id in G.nodes:
      max_neighbors = max(max_neighbors, neighbors_count(G, id))
    return max_neighbors

def shortest_path_length(G, u, v):
    path_len = 0
    try:
          path_len = len(nx.shortest_path(G, u, v))
    except:
          pass
    return path_len

def max_path_length(G):
    path_len = 0
    for u, v in G.edges:
          p = shortest_path_length(G, u, v)
          path_len = max(path_len, p)
    return path_len

class RLRouting():
    def __init__(self, snapshot, env, approach='PPO', max_steps=10, seed=48, load_from=False) -> None:
        self.snapshot = snapshot
        self.approach = approach
        self.max_steps = max_steps
        self.env = env
        self.model = None
        
        if self.approach == 'PPO':
            model_class = PPO
        else:
            model_class = None
            print(f'{approach} - not implemented!')
            raise ValueError
        
        if load_from and os.path.exists(load_from):
            self.model = model_class.load(load_from, self.env, force_reset=False)
            if seed:
                self.model.set_random_seed(seed)
            print(f'{self.approach}-model is loaded: {load_from}')
    
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
        if (not self.env) or (not self.model) or (not self.snapshot):
            print('Model is not initialized!')
            raise ValueError
        for idx, s in self.snapshot.items():
            #if nx.is_isomorphic(G, s['subgraph']):
            if u in s['subgraph'].nodes and v in s['subgraph'].nodes:
                self.g = s['subgraph']
                sample = {idx : {'subgraph' : s['subgraph'],
                                 'subgraph_transactions' : [(u, v, amount, None)]}}
                self.env.samples = sample

                start_time = time.time()        
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
            
def track_emissions(g, tx_set, routingObj, alg):
    tx_results = []
    with OfflineEmissionsTracker(country_iso_code="CAN", measure_power_secs=1, tracking_mode='process') as tracker:
        for tx in tqdm(tx_set, leave=True, desc=alg):
            u, v, amount = tx[0], tx[1], tx[2]
            tx_results.append(routingObj.routePath(g, u, v, amount))
    return tx_results

def to_bidirected(G):
    G = nx.DiGraph(G)
    for u, v in G.edges:
        if not G.has_edge(v, u):
            G.add_edge(v, u)
            attrib = {(v, u) : G.edges[u, v]}
            nx.set_edge_attributes(G, attrib)
    return G

class DijkstraRouting():
    def __init__(self, approach='dijkstra') -> None:
        self.approach = approach
        self.path = None
    
    def name(self):
        return f"Built-in Dijkstra's routing"           
        
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
            if [path[i], path[i+1]] in self.g.edges:
                total_delay += self.get_delay(path[i], path[i+1])
        return total_delay
        
    def get_total_amount(self, path, amount):
        total_amount = amount
        for i in range(1, len(path)-1):
            if [path[i], path[i+1]] in self.g.edges:
                total_amount += self.get_amount(path[i], path[i+1], total_amount)
        return total_amount

    def routePath(self, G, u, v, amount):
        r = {'path' : None, 'ok' : False}
        self.g = G
        start_time = time.time()
        try:
            self.path = nx.shortest_path(G, u, v, method=self.approach)
        except:
            pass

        if self.path:
                r["path"] = self.path
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

base_dir = './'
snapshots_dir = os.path.join(base_dir, 'snapshots')
weights_dir = os.path.join(base_dir, 'weights')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
    
algorithms = {'RLA': None,
              'LND': LNDRouting(),
              'CLN': CLightningRouting(random.uniform(-1,1)),
              'ECL': EclairRouting(),
              'A* ': DijkstraRouting(),
             }

samples = [100]#, 300, 500]
subsets = ['centralized']
versions = ['_env']
approaches = ['PPO']

for ag in range(agents):
    if os.path.exists(os.path.join(results_dir, 'emissions.csv')): 
        print('...remove old emissions.csv')
        os.remove(os.path.join(results_dir, 'emissions.csv'))

    emissions_idx = []  
    alg_results = {}
    for s in subsets:
        sample_results = {}
        for i in tqdm(samples, leave=False):
            snapshot = nx.read_gpickle(os.path.join(snapshots_dir, f'graph-sample-{i}.pickle'))
            G = snapshot['undirected_graph']
            print(f"undirected_graph, n: {len(G.nodes)} e: {len(G.edges)}")
            snapshot_ = {idx : {'subgraph' : s_['subgraph'], #here we have 50/50 testset
                                    'subgraph_transactions' : s_['subgraph_transactions'][s][1000:]}
                                for idx, s_ in snapshot['samples'].items()}
            snapshot = {idx : {'subgraph' : s_['subgraph'],
                                    'subgraph_transactions' : s_['subgraph_transactions'][s][:1000]}
                                for idx, s_ in snapshot['samples'].items()}
            if s == 'centralized':
                snapshot_ = snapshot.copy() 
            
            random.seed(48)
            for algorithm, _routingObj in tqdm(algorithms.items(), leave=False):
                    subgraph = snapshot_[ag]['subgraph'].copy()
                    tx_set = snapshot_[ag]['subgraph_transactions']
                    if algorithm == 'RLA':
                        for a in approaches:
                            for v in versions:
                                f = os.path.join(weights_dir, f'{a}-{s}-{v}-{i}-{ag}.sav')
                                best_score_file_name = sorted(glob.glob(f + '*'))
                                best_score_file_name = best_score_file_name[-1] if best_score_file_name else None
                                best_score = float(best_score_file_name.split('.sav')[1]) if best_score_file_name else 0
                                if best_score:
                                    print(f'{v:<9} {i:<3} score: {best_score:<5} {best_score_file_name}')
                                    change_env(v, best_score_file_name, snapshot_)
                                    if f"{algorithm}:{a}:{v}-{s}-{i}" not in alg_results:
                                        alg_results[f"{algorithm}:{a}:{v}-{s}-{i}"] = []
                                    alg_results[f"{algorithm}:{a}:{v}-{s}-{i}"] += track_emissions(subgraph, tx_set, routingObj, routingObj.name())
                                    emissions_idx.append(f"{algorithm}:{a}:{v}-{s}-{i}-{ag}")
                    else:
                        if f"{algorithm}-{s}-{i}" not in alg_results:
                            alg_results[f"{algorithm}-{s}-{i}"] = []
                        alg_results[f"{algorithm}-{s}-{i}"] += track_emissions(to_bidirected(subgraph), tx_set, _routingObj, _routingObj.name())
                        emissions_idx.append(f"{algorithm}-{s}-{i}-{ag}")

    if os.path.exists(os.path.join(results_dir, 'emissions.csv')): 
        e = pd.read_csv(os.path.join(results_dir, 'emissions.csv'))
        e = pd.concat([e, pd.Series(emissions_idx, name='emissions_idx')], axis=1)
        e = e[['timestamp', 'duration', 'emissions', 
            'emissions_rate', 'cpu_power', 'gpu_power', 'ram_power', 'cpu_energy',
            'gpu_energy', 'ram_energy', 'energy_consumed', 'emissions_idx']]
        alg_results['emissions'] = e.to_dict()
        os.remove(os.path.join(results_dir, 'emissions.csv'))

    with open(os.path.join(results_dir, f'results-{ag}.pickle'), 'wb') as f:
            pickle.dump(alg_results, f)

print('done')
print(f"git add -f results-*.pickle && git commit -m results && git push")