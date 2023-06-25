import os

subgraphs = [50, 60, 70, 80, 90, 100]
idxs = list(range(1, 5))
n_envs = 16
log = '#'
output = '#'
error = ''

for subgraph in subgraphs:
    for idx in idxs:
        line = f'''executable = /bin/bash 
arguments = -i conda_activate.sh baselines train.py --n_envs {n_envs} --subgraph {subgraph} --idx {idx}
transfer_input_files = conda_activate.sh, train.py 
{log}log = ./logs/train-{n_envs}-{subgraph}-{idx}.txt 
{output}output = ./logs/train-stdout-{n_envs}-{subgraph}-{idx}.txt
{error}error = ./logs/train-stderr-{n_envs}-{subgraph}-{idx}.txt
should_transfer_files = IF_NEEDED 
request_gpus = 1
queue
'''
        with open(f'run/train-{n_envs}-{subgraph}-{idx}.sub', 'wt') as file:
            file.writelines(line)