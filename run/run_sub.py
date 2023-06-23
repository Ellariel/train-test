import os

subgraphs = [50, 60, 70, 80, 90, 100]
idxs = list(range(1, 5))
#os.system('cd ..')
for subgraph in subgraphs:
    for idx in idxs:
        os.system(f'./run/train-{subgraph}-{idx}.sub')