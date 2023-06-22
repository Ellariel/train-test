from queue import  PriorityQueue
from base import Routing
from math import inf
import time

class LNDRouting(Routing):
    LND_RISK_FACTOR = 0.000000015
    A_PRIORI_PROB = 0.6

    # Initialize routing algorithm
    def __init__(self) -> None:
        super().__init__()

    # human-readable name for routing algorithm
    def name(self):
        return "LND"

    # cost function for lnd, we ignore the probability bias aspect for now.
    def cost_function(self, G, amount, u, v):
        fee = G.edges[v,u]['fee_base_sat'] + amount * G.edges[v, u]['fee_rate_sat']
        alt = (amount+fee) * G.edges[v, u]["delay"] * self.LND_RISK_FACTOR + fee
        return alt

    # cost function for first hop: sender does not take a fee
    def cost_function_no_fees(self, G, amount, u, v):
        return amount*G.edges[v,u]["delay"]*self.LND_RISK_FACTOR

    # construct route using lnd algorithm (uses ordinary dijkstra)
    def routePath(self, G, u, v, amt, payment_source=True, target_delay=0):
        r = {'path' : None, 'ok' : False}
        start_time = time.time()
        
        path, delay, amount, dist =  self.Dijkstra(G, u, v, amt, payment_source, target_delay)
            
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
