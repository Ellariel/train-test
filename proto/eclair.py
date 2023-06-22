from queue import PriorityQueue
from base import Routing
import nested_dict as nd
from math import inf
import random as rn
import requests
import time

# Retrieves current block height from API
# in case of fail, will return a default block height
def getBlockHeight():
    print("Getting block height for Eclair...")
    API_URL = "https://api.blockcypher.com/v1/btc/main"
    try:
        CBR = requests.get(API_URL).json()['height']
        print("Block Height used:", CBR)
        return CBR
    except:
        print("Block height not found, using default 697000")
        return 697000

class EclairRouting(Routing):
    CBR = getBlockHeight()

    MIN_DELAY = 9
    MAX_DELAY = 2016
    MIN_CAP = 1
    MAX_CAP = 100000000
    MIN_AGE = 505149
    MAX_AGE = CBR #update to current block
    DELAY_RATIO = 0.15
    CAPACITY_RATIO = 0.5
    AGE_RATIO = 0.35

    def __init__(self) -> None:
        super().__init__()

    def name(self):
        return "Eclair"

    def cost_function(self, G, amount, u, v):
        fee = G.edges[v,u]['fee_base_sat'] + amount * G.edges[v,u]['fee_rate_sat']
        ndelay = self.normalize(G.edges[v, u]["delay"], self.MIN_DELAY, self.MAX_DELAY)
        ncapacity = 1 - (self.normalize((G.edges[v, u]["balance_sat"] + G.edges[u, v]["balance_sat"]), self.MIN_CAP, self.MAX_CAP))
        nage = self.normalize(self.CBR - G.edges[v, u]["age"], self.MIN_AGE, self.MAX_AGE)
        alt = fee * (ndelay * self.DELAY_RATIO + ncapacity * self.CAPACITY_RATIO + nage * self.AGE_RATIO)
        return alt

    # cost function for first hop: sender does not take a fee
    def cost_function_no_fees(self, G, amount, u, v):
        return 0

    # construct route using Eclair algorithm, uses a modified general Dijkstra's algorithm
    def routePath(self, G, u, v, amt, payment_source=True, target_delay=0):
        r = {'path' : None, 'ok' : False}
        start_time = time.time()
        paths =  self.Dijkstra_general(G, u, v, amt, payment_source, target_delay)

        # fail when no paths found
        if(paths[0]==[]):
            return r

        # cut short when optimal path is direct
        if len(paths[0]) == 2:
            path = paths[0]
            delay = 0
            amount = amt
            dist = 0
        else:
            path = paths[rn.randint(0, 2)]
            delay = target_delay
            amount = amt
            dist = 0
            # recalculate based on chosen path
            for m in range(len(path) - 2, 0, -1):
                delay += G.edges[path[m], path[m + 1]]["delay"]
                amount += G.edges[path[m], path[m + 1]]["fee_base_sat"] + amount * G.edges[path[m], path[m + 1]]["fee_rate_sat"]
            delay += G.edges[path[0], path[1]]["delay"]
        
        r["runtime"] = time.time() - start_time
        r["path"] = path
        r["delay"] = delay
        r["amount"] = amount
        r["dist"]= dist
        
        r["dist"] = len(r["path"]) if r["path"] else 0
        r["ok"] = r["dist"] > 0
        r["u"] = u
        r["v"] = v
        r["feeratio"] = r["amount"] / amt
        r["feerate"] = r["amount"] / amt - 1
        r["amount"] = amt
        
        return r        

    # Generalized Dijkstra for 3 best paths - alternative to Yen's alg.
    def Dijkstra_general(self,G,source,target,amt, payment_source, target_delay):
        paths = {}
        paths1 = {}
        paths2 = {}
        dist = {}
        dist1 = {}
        dist2  = {}
        delay = {}
        delay1 = {}
        delay2 = {}
        amount = {}
        amount1 = {}
        amount2 = {}
        visited = {}
        for node in G.nodes():
            amount[node] = -1
            amount1[node] = -1
            amount2[node] = -1
            delay[node] = -1
            delay1[node] = -1
            delay2[node] = -1
            dist[node] = inf
            dist1[node] = inf
            dist2[node] = inf
            visited[node] = 0
            paths[node] = []
            paths1[node] = []
            paths2[node] = []
        prev = {}
        pq = PriorityQueue()
        dist[target] = 0
        dist1[target] = 0
        dist2[target] = 0
        delay[target] = target_delay
        delay1[target] = target_delay
        delay2[target] = target_delay
        paths[target] = [target]
        paths1[target] = [target]
        paths2[target] = [target]
        amount[target] = amt
        amount1[target] = amt
        amount2[target] = amt
        pq.put((dist[target], target))
        k = 0
        path = {}
        while 0 != pq.qsize():
            curr_cost, curr = pq.get()
            if curr_cost > dist2[curr]:
                continue
            if visited[curr] == 0:
                p = paths[curr]
                d = delay[curr]
                a = amount[curr]
                di = dist[curr]
            elif visited[curr] == 1:
                p = paths1[curr]
                d = delay1[curr]
                a = amount1[curr]
                di = dist1[curr]
            elif visited[curr] == 2:
                p = paths2[curr]
                d = delay2[curr]
                a = amount2[curr]
                di = dist2[curr]
            visited[curr]+=1
            for [v, curr] in G.in_edges(curr):
                if payment_source and v == source and G.edges[v, curr]["balance_sat"] >= a:
                    path[k]= [v] + p
                    k+=1
                    if k == 3:
                        return path
                if (G.edges[v, curr]["balance_sat"] + G.edges[curr, v]["balance_sat"] >= a) and visited[v]<3 and v not in p:
                    if (v != source or not payment_source):
                        cost = di + self.cost_function(G, a, curr, v)
                        if cost < dist[v]:
                            dist2[v] = dist1[v]
                            paths2[v] = paths1[v]
                            delay2[v] = delay1[v]
                            amount2[v] = amount1[v]
                            dist1[v] = dist[v]
                            paths1[v] = paths[v]
                            delay1[v] = delay[v]
                            amount1[v] = amount[v]
                            dist[v] = cost
                            paths[v] = [v] + p
                            delay[v] = G.edges[v, curr]["delay"] + d
                            amount[v] = a + G.edges[v, curr]["fee_base_sat"] + a * G.edges[v, curr]["fee_rate_sat"]
                            pq.put((dist[v], v))
                        elif cost < dist1[v]:
                            dist2[v] = dist1[v]
                            paths2[v] = paths1[v]
                            delay2[v] = delay1[v]
                            amount2[v] = amount1[v]
                            dist1[v] = cost
                            paths1[v] = [v] + p
                            delay1[v] = G.edges[v, curr]["delay"] + d
                            amount1[v] = a + G.edges[v, curr]["fee_base_sat"] + a * G.edges[v, curr]["fee_rate_sat"]
                            pq.put((dist1[v], v))
                        elif cost < dist2[v]:
                            dist2[v] = cost
                            paths2[v] = [v] + p
                            delay2[v] = G.edges[v, curr]["delay"] + d
                            amount2[v] = a + G.edges[v, curr]["fee_base_sat"] + a * G.edges[v, curr]["fee_rate_sat"]
                            pq.put((dist2[v], v))
        return [], -1, -1, -1

    # normalize between max and min
    def normalize(self, val, min, max):
        if val <= min:
                return 0.00001
        if val > max:
                return 0.99999
        return (val - min) / (max - min)
