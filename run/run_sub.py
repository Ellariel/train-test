import os

subgraphs = [60, 70, 80, 90, 100]
idxs = list(range(1, 5))
n_envs = 16
#os.system('cd ..')
for subgraph in subgraphs:
    for idx in idxs:
        os.system(f'condor_submit ./run/train-{n_envs}-{subgraph}-{idx}.sub')