# Phase 2: Tiled Matrix Multiplication

# Setup
- Benchmarked tiled matrix multiplication kernel vs cuBLAS (Nvidia standard)
- Testing was done on RTX 5080, CUDA 12.8.
- Ran 10 runs and took median of all execution times

- Tested matrix sizes: 2048 x 2048, 4096 x 4096
    - Reasoning
        - Established in naive kernel that workload was too small
        - TIled kernel should run faster than naive due to the reuse 
            - This would make the noise from ramp-up and tail tail effects even worse 

- Tested tile sizes: 8, 16, 32
    - Reasoning 
        - 8x8 -> 64 threads -> 2 warps full, clean for warp efficiency
        - Above tile size 32 
            - 64x64 is 4096 threads/block which is greater than the max 1024 threads/block

# Observations from Kernel Execution Times

<img width="1181" height="618" alt="image" src="https://github.com/user-attachments/assets/640f8b0b-b3cb-4896-99fd-ac019cccd9cd" />

- 2048 x 2048 matrix multiply
    - Best Performing Tile Size: 32
        - Kernel is still 33% slower than cuBLAS but 41% faster than naive kernel

- 4096 x 4096 matrix multiply
    - Best Performing Tile Size: 16
        - Kernel is 2x slower than cuBLAS but 55% faster than naive kernel
          
<img width="2223" height="987" alt="image" src="https://github.com/user-attachments/assets/d98302fc-7a35-43cf-a577-2ac45ca87391" />

- Hypothesis:
    - Speedups are due to balance in amount of reuse and number of blocks
        - Bigger tile sizes 
            - More reuse per block, less fetches from global memory
                - Tile Size 32 has 2x more reuse compared to tile size 16 and 4x of tile size 8
            - Downsides
                - Blocks have more warps, which means fewer blocks fit in 1 SM
                    - On 5080 max warps per SM is 48
                        - 32 tile size -> 32 warps/block -> 1 block/SM
                    - Less blocks for the SM to switch to in order to latency hide
    - At 2048x2048
        - All matrix data (A, B, C) is small enough that they all get cached to L2
            - Even though tile size 32 only has 1 block/SM 
                - Memory stalls don't hurt as much since pulls are from L2 
                    - Less latency to hide
    - At 4096x4096
        - Matrixes are now too big to all fit in L2
            - Memory stalls hurt more since they come from DRAM now
                - Tile size 32's 1 block/SM hurts a lot more 
                    - No other blocks to switch to for latency hiding

# Observations from Nsight Compute

- 2048 x 2048 matrix mult with tile size 32
    - Identically high compute AND memory throughput 
        - Both are being limited by the same bottleneck
    - High L1 Cache Throughput 
        - L2 Cache throughput now at 11.42%
        - Achieved what we wanted as compared to naive kernel
            - Move memory to be accessed from L1/shared compared to L2
        - Now L1/shared is the bottleneck
            - 85% throughput
            - Average 23.3 cycles being stalled waiting for Mem IO instructions
                - We are doing two loads per addition
                    - Queue fills up since we are queueing 2 loads per cycle but LSU can only process 1 load per cycle for FMA
                - Solutions:
                    - Need to reduce loads per FMA to prevent queue from filling up
    - Still 76-79% no eligible warps for scheduling, 30% SM busy
        - Warps are still waiting on memory, now the memory just comes faster from shared
        - Same issue as above with the MIO filling up
            - Because MIO queue filling up -> warps are stalled -> none available 
    - Tile Size 32 vs 16
        - Tile Size 16's compute/memory throughput is much higher 96% vs 83.58%
            - This is because smaller tile -> more blocks/SM -> more latency hiding by switching
            - However, L2 cache throughput is 2x higher
                - Smaller tile size means more blocks -> more fetches from global/cached L2 

