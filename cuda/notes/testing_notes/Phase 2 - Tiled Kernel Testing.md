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

- 2048 x 2048 matrix multiply
    - Best Performing Tile Size: 32
        - Kernel is still 33% slower than cuBLAS but 41% faster than naive kernel

- 4096 x 4096 matrix multiply
    - Best Performing Tile Size: 16
        - Kernel is 2x slower than cuBLAS but 55% faster than naive kernel

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