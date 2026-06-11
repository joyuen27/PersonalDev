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
 <img width="2541" height="900" alt="image" src="https://github.com/user-attachments/assets/33a1667a-bc8b-4f7a-9403-888688b2ca12" />

 - Thread count doesn't really affect executing timing
    - 64+ threads on 2048 x 2048 and 4096 x 4096
        - Only saw around 7-10% variation in times between different thread sizes
        - No one best thread count for all sizes
    - Having more warps per block allows scheduler to switch between to hide latency
        - In this case, all the warps might be waiting for memory, no ready warp to switch to
     
<img width="2542" height="900" alt="image" src="https://github.com/user-attachments/assets/7ffd20c5-a519-451c-9395-1a76ba2cd2c2" />

# Observations from Single Nsight
  Tested Best Performing Configurations from Kernel Execution Times, Single Run
 - 64 threads at 512 x 512
    - Good L1 Cache Throughput 80%, L2 Cache Throughput 40% low DRAM throughput 3%
        - Likely due to small matrix being able to fit mostly in L1
        - Little pulls from global memory
    - Workload is too small
        - Waves per SM - 2.03, startup + tail at the end accounts for good percentage of the kernel time
        - Too little steady-state time to benchmark kernel

 - 256 threads 2048 x 2048
    - Workload is enough to benchmark
        - Waves/SM = 32.51, assuming ramp + tail = 2 waves, have 30 waves of steady state to benchmark
    - Memory Throughput Bottleneck
        - L2 at 93.75% throughput
            - Since different threads are pulling the same data over and over again
        - 90.6% of cycles waiting on memory
            - Warps stall 48 cycles per instruction waiting on memory with 90.6% of cycles waiting on memory
                - confirms memory access (not compute) is on the critical path.
            - Nsight Recommendation: "Consider moving frequently used data to shared memory"
    - Not compute bound
        - 65% Compute Throughput
            - SM throughput of 65% sounds healthy until you check FMA pipe utilization at 8% — the SMs are busy issuing loads, not doing math.
        - "The workload achieved 4% of this device's fp32 peak performance"
    - Scheduler Starved
        - 78% of cycles, all warps are stalled waiting on memory
        - Can't just add more threads/warps

 - 512 threads at 4096 x 4096
    - Memory Throughput Bottleneck
        - L2 at 94% throughput
            - Same cause as 2048: repeated pulls of the same row/col data from L2
        - ~79% of cycles waiting on memory
            - Warps stalling on memory accesses
            - Nsight Recommendation: "Consider moving frequently used data to shared memory"
    - Not compute bound
        - ~60% SM Compute Throughput — misleading at face value
            - SM throughput of 60% sounds healthy until you check FMA pipe utilization at 8% — the SMs are busy issuing loads, not doing math.
        - Most of compute being used for memory operations: 60% pipe utilization on LSU (Load/Store Unit)
        - See roofline section below: 3% of FP32 peak achieved
    - Scheduler Starved
        - 79% of cycles, all warps are stalled waiting on memory
        - Can't just add more threads/warps
    - Sufficient workload
        - Waves/SM = 130
    - 81ms kernel time
        - This is the configuration profiled in detail in the roofline section below.
   
<img width="2061" height="1021" alt="image" src="https://github.com/user-attachments/assets/444fd3f7-153a-47d5-b9dd-14ebd1f6f6e5" />

# Roofline Analysis
Done with 512 threads at 4096 x 4096, see notes for detailed math
- Throughputs
    - Compute Throughput: 1.696 TFLOP/s
        - 3% of Peak FP32
    - DRAM Memory Throughput: 103.8 GB/s
        - 10.8% of Peak DRAM BW
    - L2 Memory Throughput: 3.41 TB/s
        - 94.2% of Peak L2 BW

- Time Floors (Theoretical times if x is the bottleneck)
    - Compute Time Floor: 2.45ms
    - DRAM Time Floor: 8.75ms
    - L2 Time Floor: 76.3ms <--- this is our bottleneck as its closest to our actual kernel time of 81.02ms
        - Using actual achieved bandwidth -> 81.0 confirms bottleneck is in L2

- Roofline Plot (L2 as binding constraint):
    - Total FLOPS/s: 1.696 TFLOPs/s
    - Arithmetic Intensity: (1.696 TFLOPs/s) / (3.41 TB/s) = 0.497 FLOPs/byte
 <img width="1739" height="1096" alt="image" src="https://github.com/user-attachments/assets/f88d87ca-d36f-4ead-9db2-d9984d30456b" />

- Takeaway
    - Bottleneck is on pulls from L2 memory as seen from the roofline analysis.
    - This is due to the fact that the naive kernel is reusing a lot of the same elements when doing matrix multiplications
    - Ex. Each row x is being multiplied against col 0 ... col y for a element. 
    - Multiple threads pull the same row/col of data over and over again from L2 cache
    - Fix. Tile computation so one pull can be reused across the whole tile.


# Phase 1: Conclusion
Phase 1 establishes: the naive kernel is L2-bandwidth-bound at 94% of peak L2 BW, achieving 3% of FP32 peak. The remaining ~96% of compute capacity is unreachable without reducing L2 traffic. Phase 2 introduces tiling to do exactly that — load tiles cooperatively into shared memory once per block.

# Phase 2: Tiled Matrix Multiplication

# Setup
- Benchmarked tiled matrix multiplication kernel vs cuBLAS (Nvidia standard)
- Testing was done on RTX 5080, CUDA 12.8.
- Ran 10 runs and took median of all execution times

- Tested matrix sizes: 512 x 512, 2048 x 2048, 4096 x 4096
- Tested tile sizes: 8, 16, 32

# Observations from Kernel Execution Times
 - Tile size 8 wins at 512 x 512
    - Again due the lower overhead of launching my kernel vs cuBLAS
    - Size 8 wins due 
