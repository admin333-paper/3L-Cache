# 3L-CACHE: Low Overhead and Precise Learning-based Eviction Policy for Web Caches

This is the implementation repository of *3L-CACHE: Low Overhead and Precise Learning-based Eviction Policy for Web Caches*. This artifact provides the source code of 3L-Cache and scripts to reproduce experiment results in our paper.

3L Cache is implemented in the [libCacheSim]([https://github.com/1a1a11a/libCacheSim]) library, and its experimental environment configuration is consistent with libCacheSim.

 ## Supported Platforms
- Software Requirements: Ubuntu 18.04

## Build and Install 
We provide some scripts for quick installation of libCacheSim.
- [Install dependency](#install-dependency)
- [Build libCacheSim](#build-libcachesim)

## Usage
After building and installing libCacheSim, cachesim should be in the _build/bin/ directory.

./bin/cachesim trace_path trace_type eviction_algo cache_size [OPTION...]
