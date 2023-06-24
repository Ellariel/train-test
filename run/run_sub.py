import os

subgraphs = [50, 60, 70, 80, 90, 100]
idxs = list(range(0, 5))
n_envs = 1
#os.system('cd ..')
for subgraph in subgraphs:
    for idx in idxs:
        os.system(f'condor_submit ./run/train-{n_envs}-{subgraph}-{idx}.sub')