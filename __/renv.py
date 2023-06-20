import random
import networkx as nx
import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces
from operator import itemgetter

from proto import cost_function

class LNEnv(Env): 
    def __init__(self, G, transactions, train=True) -> None:
        self.subset = transactions
        self.train = train
        self.g = G
        self.observation_size = len(self.g.nodes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.observation_size, ), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.observation_size + 1, ), dtype=np.float32)
        
        self.id_to_idx = {}
        self.idx_to_id = {}
        for idx, id in enumerate(self.g.nodes):
            self.id_to_idx[id] = idx
            self.idx_to_id[idx] = id
        
        self.reward = []
        self.heatmap = []

    def get_shortest_path(self, u, v, proto='dijkstra'):
        try:
            path = nx.shortest_path(self.g, u, v, method=proto)
        except:
            return
        return path  
  
    def get_path_len(self, path):
        if path:
            return len(path)
        return 100

    #def shortest_path_len(self, u, v, proto='dijkstra'):
    #    if u == v:
    #        return 0
    #    try:
    #        path_len = nx.shortest_path_length(self.g, u, v, method=proto)
    #    except:
    #        return 100
    #    return path_len
    
    def reset(self):
        tx = random.choice(self.subset)
        self.u, self.v, self.amount = tx[0], tx[1], tx[2]     
        self.path = [self.u]
        
        self.guided_path = self.get_shortest_path(self.u, self.v)
        if not self.guided_path:
            self.guided_path = [self.u, self.v]        
        
        self.agent_pos_idx = self.id_to_idx[self.u]
        self.target_pos_idx = self.id_to_idx[self.v]
        self.current_observation = np.zeros((self.observation_size, ), dtype=np.float32)
        self.current_observation[self.target_pos_idx] = 0.5
        self.current_observation[self.agent_pos_idx] = 1
        return self.current_observation 
            
    def step(self, action):
        reward = 0
        done = True
        self.current_observation = np.zeros((self.observation_size, ), dtype=np.float32)
        self.current_observation[self.target_pos_idx] = 0.5
        self.current_observation[self.agent_pos_idx] = 1
        
        direction = (action[0] + 1) / 2       
        action = action[1:]

        idx = np.nonzero(action > 0)[0]
        act = [self.idx_to_id[i] for i in idx]
        idx = np.argsort(itemgetter(*idx)(action))
        action = itemgetter(*idx)(act)

        while True:
            neighbors = [n for n in action if n in self.g.neighbors(self.path[-1]) and n not in self.path]
            if len(neighbors):
                next_node = neighbors[int(direction * (len(neighbors) - 1))]
                self.path += [next_node]
                self.current_observation[self.id_to_idx[next_node]] = 1            
            else:
                break

        if self.train:
                self.heatmap += self.path
                reward += self.compute_reward() 
                self.reward.append(reward)          
        
        return self.current_observation, reward, done, {}

    def render(self, mode='console'): 
        if mode != 'console':
          raise NotImplementedError()
        pass

    def close(self):
        pass
        
    def check_path(self):
        for i in range(0, len(self.path) - 1):
            if not self.g.has_edge(self.path[i], self.path[i + 1]):
                return False
            if self.path[i] == self.v or self.path[i + 1] == self.v:
                return True
        return False 

    def get_path_cost(self):
        total_cost = 0
        for i in range(1, len(self.path) - 1):
            if self.g.has_edge(self.path[i], self.path[i + 1]):
                if self.path[i] == self.v or self.path[i + 1] == self.v:
                    break 
                total_cost += cost_function(self.g, self.path[i], self.path[i + 1], self.amount)
        return -total_cost

    def get_guided_bonus(self):
        distance = 0
        for i in self.path[1:]:
            for j in self.guided_path:
                distance += self.get_path_len(self.get_shortest_path(i, j))
        distance -= len(self.path)
        return -distance
   
    def compute_reward(self):
        reward = 0
        if self.check_path():
            path = self.get_path()
            reward += 10000 * len(self.guided_path) / len(path)
            # reward += (self.get_path_cost() / 1000) / len(path)
        reward += self.get_guided_bonus()
        return reward
        
    def get_path(self):
        if self.v in self.path:
            return self.path[:self.path.index(self.v) + 1]
        return self.path
        
    def get_reward(self):
        return self.reward
        
    def get_heatmap(self):
        unique, counts = np.unique(self.heatmap, return_counts=True)
        return np.asarray((unique, counts)).T