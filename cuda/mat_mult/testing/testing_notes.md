# Phase 1: Base Matrix Multiplication Testing

# Setup
- Benchmarked naive matrix multiplication kernel vs cuBLAS (Nvidia standard)
- Testing was done on RTX 5080, CUDA 12.8.
- Ran 10 runs and took median of all execution times

- Tested matrix sizes: 512 x 512, 2048 x 2048, 4096 x 4096 
- Tested thread counts: 32, 64, 128, 256, 512, 1024

# Observations from Kernel Execution Times
 - Kernel actually beats cuBLAS at 512 x 512 matrix
    - Base is more lightweight, less initialization so better at smaller matrixes
    - cuBLAS has some initialization overhead, had to warm it up
 - Execution time around
    - 50% of cuBLAS at 2048 x 2048
    - 25-30% of cuBlas at 4096 x 4096
 - Thread count doesn't really affect executing timing
    - 64+ threads on 2048 x 2048 and 4096 x 4096
        - Only saw around 7-10% variation in times between different thread sizes
        - No one best thread count for all sizes
    - Having more warps per block allows scheduler to switch between to hide latency
        - In this case, all the warps might be waiting for memory, no ready warp to switch to

# Observations from Single Nsight
  Tested Best Performing Configurations from Kernel Execution Times, Single Run

 - 64 threads at 512 x 512
    - Good L1 Cache Throughput 80%, L2 Cahce Throughput 40% low DRAM throughput 3%
        - Likely due to small matrix being able to fit mostly in L1
        - Little pulls from global memory
    - Workload is too small
        - Only 30% SM busy

 - 256 threads 2048 x 2048
    - Memory Throughput Bottleneck
        - L2 at 93.75% throughput
            - Since we are pulling the same data over and over again
                - Ie. Pulling the same col again every time for each multiplication
        - 90.6% of cycles waiting on memory
            - Warps stall 48 cycles per instruction waiting on memory
            - Nsight Recommendation: "Consider moving frequently used data to shared memory"
    - Not compute bound
        - 65% Compute Throughput
        - "The workload achieved 4% of this device's fp32 peak performance"
    - Scheduler Starved
        - 78% of cycles, all warps are stalled waiting on memory
        - Can't just add more threads/warps

 - 512 threads at 4096 x 4096
    - Same notes as above
        - L2 being slammed 94% throughput
        - 79% of cycles no eligible warps
        - Most of compute being used for memory operations: 60% pipe utilization on LSU (Load/Store Unit)


# Phase 2: Tiled Matrix Multiplication

# Setup