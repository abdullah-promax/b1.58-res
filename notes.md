**To Run This Code:**

1.  Save each block of code into its respective `.py` file (e.g., `quant_utils.py`, `rms_norm.py`, `bit_matmul_kernel.py`, `bit_linear_layer.py`, `cifar10_model.py`, `train_cifar10.py`).
2.  Make sure all files are in the same directory, or adjust Python's import paths.
3.  Ensure you have PyTorch, Torchvision, and Triton installed in your Python environment.
4.  Run the main training script: `python train_cifar10.py`

**Potential Issues & Next Steps:**

*   **Triton Kernel Debugging:** The Triton kernel is complex. If it has errors:
    *   Start with very small `M, N, K` values.
    *   Use `tl.device_print` inside the kernel (be sparse with it).
    *   Compile with `triton.compile(..., "kernel_name.ttgir")` and inspect the intermediate representation.
    *   Compare its output with a PyTorch equivalent of the packed matmul for small tensors.
*   **BLOCK\_SIZE\_K vs Packing:** The Triton kernel `bit_matmul_kernel` has `BLOCK_SIZE_K` which refers to the *unpacked* K dimension processed by a block. The current autotune configs might need adjustment to ensure `BLOCK_SIZE_K` is always a multiple of 4. A `static_assert` or runtime check in the Python wrapper for `bit_matmul_kernel` call would be good. My kernel logic implicitly expects this for the inner B unpacking loop.
*   **Performance:** The current Triton kernel is a straightforward implementation. Optimizations like using shared memory more effectively for `a_tile` and `b_unpacked_tile` (if it were fully formed), instruction reordering, and careful choice of `num_warps` and `num_stages` can significantly impact performance. The current unpacking of B "on-the-fly" for each 4-row segment might not be the most performant way compared to unpacking a larger `B` tile into shared memory first.
*   **Numerical Stability:** Quantization Aware Training (QAT) with very low bits can be unstable. Learning rates, weight decay, and initialization are crucial. The dequantization factors `A_scale * B_ternary_scale` could become very small or very large, leading to NaNs or Infs. Adding epsilon and clipping might be necessary. The `GradScaler` helps with mixed precision stability.
*   **Gradient Correctness:** Double-check the STE implementation in `BitMatmulFunction.backward`. Errors here can prevent the model from learning.
*   **Autotuner:** The autotuner configs provided are basic. More diverse configs, especially considering the `BLOCK_SIZE_K % 4 == 0` constraint, would be beneficial.

This comprehensive set of files should give you a strong starting point for your 1.58-bit CIFAR-10 experiment with Triton! Good luck!
