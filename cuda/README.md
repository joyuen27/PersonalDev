> **Disclaimer**
> This README was drafted with help from AI and should be updated as the project evolves.

# CUDA Practice

This folder tracks my CUDA learning and practice so far.

## What I have done

- Started a CUDA matrix multiplication exercise in `mat_mult/`.
- Wrote a baseline GPU kernel where each CUDA thread computes one output element.
- Added host-side code for matrix dimension checks, GPU memory allocation, host/device copies, kernel launch, synchronization, and cleanup.
- Built a simple correctness check using a `512 x 512` identity matrix multiplied by another matrix.
- Added CUDA learning notes in `notes/cuda_notes.pdf`, titled `CUDA Programming Notes`.

## Files

- `mat_mult/mat_mul_base.cu` - baseline matrix multiplication implementation.
- `mat_mult/prac_dev.cu` - practice copy of the matrix multiplication implementation.
- `notes/cuda_notes.pdf` - 4-page CUDA programming notes PDF.

## Current status

The matrix multiplication code is close to a first working CUDA example, but there is one cleanup needed before compiling: the kernel is named `k_mat_mul_base`, while the launch currently calls `k_mat_mul`.

## Next steps

- Fix the kernel name mismatch.
- Add CUDA error checks after memory allocation, memory copies, kernel launch, and synchronization.
- Compile and run the matrix multiplication example with `nvcc`.
- Keep improving the notes as I learn more about CUDA threads, blocks, grids, memory, and performance.
