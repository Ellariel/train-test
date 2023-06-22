from queue import PriorityQueue
from math import inf
import nested_dict as nd


class Routing:
    def __init__(self):
        pass

    # Dijkstra's routing algorithm for finding the shortest path
    def Dijkstra(self, G, source, target, amt, payment_source=True, target_delay=0):
        paths = {}
        dist = {}
        delay = {}
        amount = {}
        for node in G.nodes():
            amount[node] = -1
            delay[node] = -1
            dist[node] = inf    
        visited = set()
        pq = PriorityQueue()
        dist[target] = 0
        delay[target] = target_delay
        paths[target] = [target]
        amount[target] = amt
        pq.put((dist[target], target))
        while 0 != pq.qsize():
            curr_cost, curr = pq.get()
            if curr == source:
                return paths[curr], delay[curr], amount[curr], dist[curr]
            if curr_cost > dist[curr]:
                continue
            visited.add(curr)
            for [v, curr] in G.in_edges(curr): # for [v, curr] in G.edges(curr): #
                if payment_source and v == source and G.edges[v, curr]["balance_sat"] >= amount[curr]:
                    cost = dist[curr] + self.cost_function_no_fees(G, amount[curr], curr, v)
                    if cost < dist[v]:
                        dist[v] = cost
                        paths[v] = [v] + paths[curr]
                        delay[v] = G.edges[v, curr]["delay"] + delay[curr]
                        amount[v] = amount[curr]
                        pq.put((dist[v], v))
                if (G.edges[v, curr]["balance_sat"] + G.edges[curr, v]["balance_sat"] >= amount[curr]) and v not in visited:
                    if (v != source or not payment_source):
                        cost = dist[curr] + self.cost_function(G, amount[curr], curr, v)
                        if cost < dist[v]:
                            dist[v] = cost
                            paths[v] = [v] + paths[curr]
                            delay[v] = G.edges[v, curr]["delay"] + delay[curr]
                            amount[v] = amount[curr] + G.edges[v, curr]["fee_base_sat"] + \
                                amount[curr]*G.edges[v, curr]["fee_rate_sat"]
                            pq.put((dist[v], v))
        return [], -1, -1, -1