import subprocess
import os
import sys
from datetime import datetime
import json
import concurrent.futures
import csv

def lib_command_executor(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

def parallel_executor_libcachesim(params):
    num_process = params['num_process']
    algo = params['algo']
    tracetype = params['type']
    other_param = '-t "time-col=1, obj-id-col=2, obj-size-col=3, delimiter=\t, has-header=false"'
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_process) as executor:
        for file, csize in zip(params['file'], params['cachesize']):
            # Running multiple cache strategies with one instruction can simultaneously return its object miss ratio and byte miss ratio
            size = ','.join([str(s) for s in csize])
            if tracetype == 'csv':
                command = command = f'./libCacheSim/libCacheSim/_build/bin/cachesim {file} {tracetype} {algo} {size} {other_param}'
            else:
                command = command = f'./libCacheSim/libCacheSim/_build/bin/cachesim {file} {tracetype} {algo} {size}'
            executor.submit(lib_command_executor, command)

if __name__ == "__main__":
        # Path of the datasets
    dataset_path = sys.argv[0]
    # Number of processes
    num_process = sys.argv[1]
    # policy
    algo = sys.argv[2]
    # Record the unique bytes of traces
    with open("./trace_unique_bytes", "r") as file:
        trace_info = json.loads(file.read())
    file_list = []
    csizes = []
    # for dataset_path in dataset_paths:
    for file in os.listdir(dataset_path):
        # only csv
        if not os.path.isdir(os.path.join(dataset_path, file)) and file.split('.')[-1] == 'csv':
            # small cache size and large cache size
            csizes.append([trace_info[file] // 1000, trace_info[file] // 10])
            file_list.append(dataset_path + file)
    
    # cache_polices = ['fifo', 'lru', 'lecar', 'lhd', 'sieve', '3lcache']
    # for cache_policy in algo:
    params = {
        'file': file_list,
        'cachesize': csizes,
        'algo': algo,
        'num_process': num_process,
        'type': 'csv'
    }
    parallel_executor_libcachesim(params)