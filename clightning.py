from queue import PriorityQueue
from base import Routing
import nested_dict as nd
from math import inf
import time

class CLightningRouting(Routing):
    C_RISK_FACTOR = 10
    RISK_BIAS = 1
    DEFAULT_FUZZ = 0.05

    def __init__(self, fuzz) -> None:
        super().__init__()
        self.__fuzz = fuzz

    def name(self):
        return "C-Lightning"

    def cost_function(self, G, amount, u, v):
        scale = 1 + self.DEFAULT_FUZZ * self.__fuzz
        fee = scale * (G.edges[v, u]['fee_base_sat'] + amount * G.edges[v, u]['fee_rate_sat'])
        alt = ((amount + fee) * G.edges[v, u]["delay"]
               * self.C_RISK_FACTOR + self.RISK_BIAS)
        return alt

    # cost function for first hop: sender does not take a fee
    def cost_function_no_fees(self, G, amount, u, v):
        return amount * G.edges[v, u]["delay"] * self.C_RISK_FACTOR + self.RISK_BIAS

    # construct route using C-lightning algorithm (uses ordinary dijkstra)
    def routePath(self, G, u, v, amt, payment_source=True, target_delay = 0 ):
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
