Of course. Here is a revised `README.md` with a more structured and visually organized layout.

-----

# 1.58-bit Linear Layer with Triton Kernel 🚀

A PyTorch implementation of a 1.58-bit (ternary) `BitLinear` layer, featuring a custom high-performance Triton kernel for quantized matrix multiplication. This project demonstrates the core concepts of the BitNet architecture on a simple CNN trained on the CIFAR-10 dataset.

-----

## ✨ Key Features

  * [cite\_start]**Custom `BitLinear` Layer**: A `nn.Module` that serves as a drop-in replacement for a standard `nn.Linear` layer, built with a custom `torch.autograd.Function` to manage the quantization workflow[cite: 16, 22].
  * [cite\_start]**1.58-bit Weight Quantization**: Full-precision weights are quantized to ternary values `{-1, 0, 1}` using the absolute mean (`absmean`) of the weight tensor for scaling[cite: 3, 62].
  * [cite\_start]**8-bit Activation Quantization**: Input activations are quantized to `int8` integers using the absolute maximum (`absmax`) value for scaling[cite: 64].
  * [cite\_start]**High-Performance Triton Kernel**: A custom `bit_matmul_kernel` written in Triton performs the core matrix multiplication on the quantized and packed tensors[cite: 6].
  * [cite\_start]**On-the-Fly Weight Unpacking**: To save memory, four 2-bit ternary weight representations are packed into a single `int8` byte[cite: 65, 66]. [cite\_start]The Triton kernel unpacks these weights on-the-fly during computation[cite: 29, 32, 33].
  * [cite\_start]**RMSNorm Integration**: Includes a standard `RMSNorm` layer to normalize the input activations before quantization, a common practice in modern network architectures[cite: 19, 74].

-----

## ⚙️ How It Works

The `BitLinear` layer executes a custom forward and backward pass for training:

1.  [cite\_start]**Normalization**: Input activations are first normalized using `RMSNorm`[cite: 19].
2.  [cite\_start]**Quantization**: Activations are quantized to `int8` and weights are quantized to ternary values `{-1, 0, 1}`[cite: 2, 3].
3.  [cite\_start]**Weight Packing**: The ternary weights are packed so that four values fit into a single `int8` byte[cite: 5, 65].
4.  [cite\_start]**Triton Execution**: The custom Triton kernel computes the matrix product of the `int8` activations and the packed ternary weights[cite: 6].
5.  [cite\_start]**Dequantization**: The output is scaled back to a floating-point representation using the quantization scales[cite: 11].
6.  [cite\_start]**Backward Pass**: A Straight-Through Estimator (STE) is used, where gradients are calculated based on the approximated full-precision tensors[cite: 14].

-----

## 📂 Repository Structure

```
.
[cite_start]├── train_cifar10.py     # Main script to train the CNN on CIFAR-10 [cite: 44]
[cite_start]├── cifar10_model.py     # Defines the SimpleBitNetCNN model architecture [cite: 37]
[cite_start]├── bit_linear_layer.py  # The core BitLinear layer and custom autograd function [cite: 1]
[cite_start]├── bit_matmul_kernel.py # The custom Triton kernel for matmul [cite: 6]
[cite_start]├── quant_utils.py       # Helper functions for quantization and packing [cite: 61]
[cite_start]├── rms_norm.py          # RMSNorm layer implementation [cite: 74]
[cite_start]└── notes.md             # Development notes and potential improvements [cite: 42]
```

-----

## 🛠️ Getting Started

### Prerequisites

Ensure you have the following libraries installed. The code is designed to be run on a CUDA-enabled GPU.

  * PyTorch
  * Torchvision
  * Triton (`pip install triton`)

### Running the Training

1.  [cite\_start]Place all `.py` files in the same directory[cite: 42].
2.  Run the main training script from your terminal:
    ```bash
    python train_cifar10.py
    ```
    [cite\_start]The script will download the CIFAR-10 dataset, initialize the model, and begin the training and evaluation process[cite: 44, 77].

-----

## 📝 Notes & Future Work

  * [cite\_start]**Kernel Performance**: The Triton kernel is a straightforward implementation[cite: 52]. [cite\_start]Performance could be improved by using shared memory more effectively or exploring different unpacking strategies[cite: 53, 54].
  * [cite\_start]**Numerical Stability**: Low-bit Quantization Aware Training (QAT) can be unstable[cite: 55]. [cite\_start]Learning rates and weight decay are crucial, and `GradScaler` is used to help with mixed-precision stability[cite: 57].
  * [cite\_start]**Autotuner**: The provided Triton autotuner configurations are basic[cite: 59]. [cite\_start]More diverse configs could lead to better performance on different hardware[cite: 60].
