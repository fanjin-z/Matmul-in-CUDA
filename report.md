# Assignment 2 Report

## Program Analysis
### Tiling
I apply tiling to reduce access to the global memory. Shared memory arrays for A and B are designated in every block. In my test, the optimal tile size for N=1024 and above is 128 by 16 and optimal tile size for 512 and below is 64 by 16. For generic purpose and overall performance, the final submission tile size is 128 by 16.

#### Register blocking
I apply register memory level blocking to reduce access to shared memory. That is, each thread will handle more than 1 work. In order to do so, register memory arrays for A, B and their calculation results are designated in every thread. In my test, 64 works per thread (8 in each dimension) is the optimal. This method contributes the most significant performance boost.


### Other tricks
#### `__ldg` intrinsic
Loading with `__ldg` will cache data to Readonly data cache. Compiler may not always interpret normal load to `__ldg`,  it will behave similar to using both `const` and `restrict` according to answer by a stackoverflow user . My test shows using `__ldg` improves a lot.

#### Configure L1 cache
Setting cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual) can adjust the ratio between L1 cache size and shared memory. However, my test doesn't show any visible improvement.

#### Change bank size to 8 bytes.
Setting cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) can change bank memroy size to 8 bytes. Because our matrix multiplication mainly operate on 8 bytes double data, memory transfer for a double type data may only need 1 instruction instead of 2. Also, it may reduce the risk of bank conflict according to a test by a stackoverflow user. However, my test doesn't show any visible improvement.

#### Avoid bank conflict
Simply padding shared memory array by 1 column, ie Bs[N][N] -> Bs[N][N+1]. This would effectively eliminate bank conflict.

## Results
The test is performed on AWS.
### Performance
#### Block Size 1
Block size = 128 by 16, work size = 8

n     |Results(GFlops)
---|---
256   |151.692738
512   |421.353956
1024  |637.437741
2048  |762.936704

#### Block Size 2
Block size = 64 by 16. work size = 4

n     |Results(GFlops)
---|---
256   |286.342804
512   |554.000082
1024  |565.072153
2048  |614.471780

#### Block Size 3
Block size = 32 by 32, work size = 4

n     |Results(GFlops)
---|---
256   |291.297529
512   |343.220311
1024  |395.255774
2048  |408.560172

#### Explaination
The choice of optimal block size is based on field test results. Generally, we want to utilize as much as registers and shared memory space as possible. that is why Block size = 128 by 16, work size = 8 win in most case. But for the small matrix, smaller block size can utilize more parallelism.

Generally, larger matrix sizes have higher performance. First, the overhead cost for large matrix is small, the cost of parallelism is insignificant compared to overall time. Second, large matrix can be more beneficial from my program design and fully utilize tiling and register blocking.

### Comparison with naive implementation

n     |Naive(GFlops)   |My-Results(GFlops)
---|---|---
256   |63.186352     |291.297529
512   |66.821841     |554.000082
1024  |82.674949     |637.437741
2048  |762.936704    |762.936704

### Comparison with BLAS

n     |BLAS(GFlops)   |My-Results(GFlops)
---|---|---
256   |5.84          |151
512   |17.4          |421
384| |430
512| |553
640| |602
768   |45.3          |675
896
1152
1023  |73.7          |
1024  |73.6          |637
1025  |73.5          |
1280| |645
1408| |655
1536| |680
1664| |692
1792| |705
1920| |740
2047  |171           |
2048  |182           |762
2049  |175           |


![speedup Ratio](speedup.png)


## Potential Future work
1. Try to increase instruction-level parallelism, e.g. more manual unrolling.
2. Try to eliminate register memory bank conflict
3. Experiment more tile size on different levels.
4. Consider L2 cache blocking.
5. Experiment on multi-gpu parallelism.


## References
CUDA Tutorial
https://cuda-tutorial.readthedocs.io/en/latest/tutorials

CUDA tutorial
http://supercomputingblog.com/cuda-tutorials/

CUDA C Programming Guide
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

NVIDIA CUDA Runtime API
https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

An Efficient Matrix Transpose in CUDA C/C++
https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/

Using Shared Memory in CUDA C/C++
https://devblogs.nvidia.com/using-shared-memory-cuda-cc/

How to Access Global Memory Efficiently in CUDA C/C++ Kernels
https://devblogs.nvidia.com/how-access-global-memory-efficiently-cuda-c-kernels/

CUDA Pro Tip: nvprof is Your Handy Universal GPU Profiler
https://devblogs.nvidia.com/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/

Using CUDA Warp-Level Primitives
https://devblogs.nvidia.com/using-cuda-warp-level-primitives/

Shuffle: Tips and Tricks
http://on-demand.gputechconf.com/gtc/2013/presentations/S3174-Kepler-Shuffle-Tips-Tricks.pdf

What is the difference between `__ldg()` intrinsic and a normal execution?
https://stackoverflow.com/a/26603851/7329044

CUDA shared memory bank conflicts report higher
https://stackoverflow.com/a/30426100/7329044
