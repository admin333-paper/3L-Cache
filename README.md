# 3L-CACHE: Low Overhead and Precise Learning-based Eviction Policy for Web Caches

This is the implementation repository of *3L-CACHE: Low Overhead and Precise Learning-based Eviction Policy for Web Caches*. This artifact provides the source code of 3L-Cache and scripts to reproduce experiment results in our paper.

#### 3L Cache is implemented in the [libCacheSim](https://github.com/1a1a11a/libCacheSim) library, and its experimental environment configuration is consistent with libCacheSim.

 ## Supported Platforms
- Software Requirements: Ubuntu 18.04

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


