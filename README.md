

# 1.58-bit Linear Layer with Triton Kernel 🚀

A PyTorch implementation of a 1.58-bit (ternary) `BitLinear` layer, featuring a custom high-performance Triton kernel for quantized matrix multiplication. This project demonstrates the core concepts of the BitNet architecture on a simple CNN trained on the CIFAR-10 dataset.

-----

## ✨ Key Features

  * **Custom `BitLinear` Layer**: A `nn.Module` that serves as a drop-in replacement for a standard `nn.Linear` layer, built with a custom `torch.autograd.Function` to manage the quantization workflow.
  * **1.58-bit Weight Quantization**: Full-precision weights are quantized to ternary values `{-1, 0, 1}` using the absolute mean (`absmean`) of the weight tensor for scaling.
  * **8-bit Activation Quantization**: Input activations are quantized to `int8` integers using the absolute maximum (`absmax`) value for scaling.
  * **High-Performance Triton Kernel**: A custom `bit_matmul_kernel` written in Triton performs the core matrix multiplication on the quantized and packed tensors.
  * **On-the-Fly Weight Unpacking**: To save memory, four 2-bit ternary weight representations are packed into a single `int8` byte. The Triton kernel unpacks these weights on-the-fly during computation.
  * **RMSNorm Integration**: Includes a standard `RMSNorm` layer to normalize the input activations before quantization, a common practice in modern network architectures.

-----

## ⚙️ How It Works

The `BitLinear` layer executes a custom forward and backward pass for training:

1.  **Normalization**: Input activations are first normalized using `RMSNorm`.
2.  **Quantization**: Activations are quantized to `int8` and weights are quantized to ternary values `{-1, 0, 1}`.
3.  **Weight Packing**: The ternary weights are packed so that four values fit into a single `int8` byte.
4.  **Triton Execution**: The custom Triton kernel computes the matrix product of the `int8` activations and the packed ternary weights.
5.  **Dequantization**: The output is scaled back to a floating-point representation using the quantization scales.
6.  **Backward Pass**: A Straight-Through Estimator (STE) is used, where gradients are calculated based on the approximated full-precision tensors.

-----

## 🛠️ Getting Started

### Prerequisites

Ensure you have the following libraries installed. The code is designed to be run on a CUDA-enabled GPU.

  * PyTorch
  * Torchvision
  * Triton (`pip install triton`)

### Running the Training

1.  Place all `.py` files in the same directory.
2.  Run the main training script from your terminal:
    ```bash
    python train_cifar10.py
    ```
    The script will download the CIFAR-10 dataset, initialize the model, and begin the training and evaluation process.

-----

## 📝 Notes & Future Work

  * **Kernel Performance**: The Triton kernel is a straightforward implementation. Performance could be improved by using shared memory more effectively or exploring different unpacking strategies.
  * **Numerical Stability**: Low-bit Quantization Aware Training (QAT) can be unstable. Learning rates and weight decay are crucial, and `GradScaler` is used to help with mixed-precision stability.
  * **Autotuner**: The provided Triton autotuner configurations are basic. More diverse configs could lead to better performance on different hardware.
