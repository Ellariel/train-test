import networkx as nx
import numpy as np
import requests, random

def normalize(x, min, max):
    if x <= min:
        return 0.0
    if x > max:
        return 0.99999
    return (x - min) / (max - min)

# Retrieves current block height from API
# in case of fail, will return a default block height
def getBlockHeight(default=True):
    if default:
        return 697000
    API_URL = "https://api.blockcypher.com/v1/btc/main"
    try:
        CBR = requests.get(API_URL).json()['height']
        print("Block height used:", CBR)
        return CBR
    except:
        print("Block height not found, using default 697000")
        return 697000

### GENERAL
BASE_TIMESTAMP = 1681234596.2736187
BLOCK_HEIGHT = getBlockHeight()
### LND
LND_RISK_FACTOR = 0.000000015
A_PRIORI_PROB = 0.6
### ECL
MIN_AGE = 505149
MAX_AGE = BLOCK_HEIGHT
MIN_DELAY = 9
MAX_DELAY = 2016
MIN_CAP = 1
MAX_CAP = 100000000
DELAY_RATIO = 0.15
CAPACITY_RATIO = 0.5
AGE_RATIO = 0.35
### CLN
C_RISK_FACTOR = 10
RISK_BIAS = 1
DEFAULT_FUZZ = 0.05
FUZZ = random.uniform(-1, 1)

def cost_function(G, u, v, amount, proto_type='LND'):
    fee = G.edges[u, v]['fee_base_sat'] + amount * G.edges[u, v]['fee_rate_sat']
    if proto_type == 'LND':
        cost = (amount + fee) * G.edges[u, v]['delay'] * LND_RISK_FACTOR + fee #+ calc_bias(G.edges[u, v]['last_failure'])*1e6
    elif proto_type == 'ECL':
        n_capacity = 1 - (normalize(G.edges[u, v]['capacity_sat'], MIN_CAP, MAX_CAP))
        n_age = normalize(BLOCK_HEIGHT - G.edges[u, v]['age'], MIN_AGE, MAX_AGE)
        n_delay = normalize(G.edges[u, v]['delay'], MIN_DELAY, MAX_DELAY)
        cost = fee * (n_delay * DELAY_RATIO + n_capacity * CAPACITY_RATIO + n_age * AGE_RATIO)     
    elif proto_type == 'CLN':
        fee = fee * (1 + DEFAULT_FUZZ * FUZZ)
        cost = (amount + fee) * G.edges[u, v]['delay'] * C_RISK_FACTOR + RISK_BIAS       
    else:
        cost = 1
    return cost