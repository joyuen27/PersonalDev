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

- 2048 x 2048 Tile Size 32 vs 16
    - Tile Size 16's compute/memory throughput is much higher 96% vs 83.58%
        - This is because smaller tile -> more blocks/SM -> more latency hiding by switching
        - However, L2 cache throughput is 2x higher
            - Smaller tile size means 2x less reuse-> 2x more fetches from global/cached L2 
    - Tradeoff between latency hiding and fetches from L2

- 4096 x 4096 matrix mult with tile size 16
    - Compute and Memory Throughput both at 96.37%
        - L1 Cache Throughput at 97.24%
            - This is the LSU pipeline again
    - SM only busy 35% of time - only 24% of warps eligible every cycle
        - Same as above cycles are being stalled on MIO instruction queues
            - "Each warp of this workload spends 26.0 cycles being stalled waiting for the MIO"

- 4096 x 4096 Tile Size 32 vs 16
    - Again we see that compute/memory throughputs are 15% higher on tile size 16
        - More latency switching as said before
    - L2 throughput is 12% higher on tile size 16
        - Less reuse -> more fetches again from global memory except they land in L2
    - At bigger matrix sizes like 4096
        - More total work per SM -> latency hiding matters more
            - Even though tile size 16 has half the reuse and 2x more pulls from global/L2

# Roofline Analysis
Done with 4096 x 4096 and tile size 16, see notes for more details

- Compute Throughput: 3.12 TFLOPS
    - 5.6% of peak fp32
    - Time Floor: 2.45ms
- L1 Throughput: 10.25 TB/s
    - 53% of peak BW
    - Time Floor: 23.1 ms
- L2 Throughput: 838.6 GB/s
    - 24.31% of peak BW
    - Time Floor: 10.7ms
- DRAM Throughput: 98.6 GB/s
    - 13.62% of peak BW
    - Time Floor: 4.52ms
  
<img width="1901" height="1103" alt="image" src="https://github.com/user-attachments/assets/cefbded7-598c-4f98-bd4e-67d8a497c693" />

- Roofline Plot
<img width="1899" height="1176" alt="image" src="https://github.com/user-attachments/assets/a7dfb652-e514-4a9d-b633-1c4825b67a9c" />

- Observations
    - Noticeable gains over naive
        - Compute throughput: 3.0% → 5.6% of FP32 peak (1.84× higher)
        - L1/TEX throughput: 66% → 97% (LSU pipe now saturated)
        - L2 throughput: 94% → 24% (4× drop — bottleneck eliminated)
        - DRAM throughput: 11% → 14% (roughly unchanged)
    - L1 throughput is only at 53% of peak BW even though NCU reported 97% L1 cache throughput
        - Bottleneck is in the LSU unit which is at 97% of peak instructions/cycle
        - This is also why none of the time floors match up with our runtime of 44ms
            - Bottleneck isn't in any of the memory/compute throughputs
    - To further optimize kernel
        - Reduce LSU instructions per FMA
            - Drain the MIO queue faster

# Conclusion

The tiled kernel design was effective in improving the L2 bottleneck seen in the naive kernel. By tiling the kernel and maximizing reuse through shared memory we are now hitting faster L1 cache leading to a higher throughput overall. The bottleneck has now shifted to the LSU pipeline where because we are issuing two load instructions to the LSU and the pipeline can only process one load per cycle, we end up filling the queue and waiting on it to empty. The next steps would be to reduce the load-to-FMA ratio in the inner loop.
