# 3L-Cache: Low Overhead and Precise Learning-based Eviction Policy for Web Caches

This is the implementation repository of *3L-CACHE: Low Overhead and Precise Learning-based Eviction Policy for Web Caches*. This artifact provides the source code of 3L-Cache and scripts to reproduce experiment results in our paper.

#### 3L Cache is implemented in the [libCacheSim](https://github.com/1a1a11a/libCacheSim) library, and its experimental environment configuration is consistent with libCacheSim.

 ## Supported Platforms
- Software Requirements: Ubuntu 18.04, cmake 3.28.6

## Trace Format
Request traces are expected to be in a space-separated format with 3 columns.
- time should be a long long int
- id should be a long long int
- size should be uint32

| time |  id | size |
| ---- | --- | ---- |
|   1  |  1  |  120 |
|   2  |  2  |   64 |
|   3  |  2  |   64 |

## Build and Install 
We provide some scripts for quick installation of libCacheSim.
```bash
cd scripts && bash install_dependency.sh && bash install_libcachesim.sh
```

## Usage
After building and installing libCacheSim, cachesim should be in the _build/bin/ directory.
```bash
~/libCacheSim/_build/bin/cachesim trace_path trace_type eviction_algo cache_size [OPTION...]
```
## Run a single cache simulation
```bash
~/libCacheSim/_build/bin/cachesim /path/to/tencentBlock_ns3964.csv csv 3lcache 1347453593  -t "time-col=1, obj-id-col=2, obj-size-col=3"
```

## Run multiple cache simulations
```bash
~/libCacheSim/_build/bin/cachesim /path/to/tencentBlock_ns3964.csv csv 3lcache 1347453593,13474535  -t "time-col=1, obj-id-col=2, obj-size-col=3"
```

## Examples
```bash

~/libCacheSim/_build/bin/cachesim /path/to/tencentBlock_ns3964.csv csv 3lcache 1347453593  -t "time-col=1, obj-id-col=2, obj-size-col=3"
# Output object miss ratio and throughput
/data/trace/tencentblock/tencentBlock_ns3964.csv 3L-Cache cache size     1GiB,         13625211 req, miss ratio 0.3473, throughput 0.58 MQPS


~/libCacheSim/_build/bin/cachesim /path/to/tencentBlock_ns3964.csv csv 3lcache 1347453593,13474535  -t "time-col=1, obj-id-col=2, obj-size-col=3"
#Output object miss ratio and byte miss ratio
result/tencentBlock_ns3964.csv                          3L-Cache cache size        1GiB, 13625211 req, miss ratio 0.3473, byte miss ratio 0.1030
result/tencentBlock_ns3964.csv                          3L-Cache cache size        0GiB, 13625211 req, miss ratio 0.5297, byte miss ratio 0.6389
```

## Evaluate algorithms on a large number of traces
```bash
cd experiment/scripts
python3 miss_ratio_simulation.py <dataset_path> <num_process> <eviction_algo>
```
The miss_ratio_simulation script can evaluate algorithms on a large number of traces, and the evaluation results will be stored in experiments/scripts/results.
