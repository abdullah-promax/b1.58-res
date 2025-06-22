import torch
import torch.nn as nn
from torch.autograd import Function
import math

from quant_utils import quantize_ternary_weights_absmean, quantize_activations_absmax, pack_ternary_weights_to_int8
from rms_norm import RMSNorm
# Import the kernel (assuming the file is in the same directory or Python path)
from bit_matmul_kernel import bit_matmul_kernel


class BitMatmulFunction(Function):
    @staticmethod
    def forward(ctx, A_fp, B_fp, K_original):
        # A_fp: (M, K_original) activations (float)
        # B_fp: (K_original, N) weights (float)
        
        M, K_orig = A_fp.shape
        _, N = B_fp.shape

        # 1. Quantize Activations (A_fp) to int8
        A_q, A_scale = quantize_activations_absmax(A_fp) # A_q is int8

        # 2. Quantize Weights (B_fp) to ternary {-1, 0, 1} (int8)
        B_ternary_unpacked, B_ternary_scale = quantize_ternary_weights_absmean(B_fp) # B_ternary_unpacked is int8

        # 3. Pad K dimension to be multiple of 4 for packing B and for A
        K_padded = math.ceil(K_orig / 4.0) * 4
        
        A_q_padded = A_q
        if K_orig != K_padded:
            pad_size = K_padded - K_orig
            A_q_padded = torch.nn.functional.pad(A_q, (0, pad_size), "constant", 0) # Pad last dim

        B_ternary_padded = B_ternary_unpacked
        if K_orig != K_padded:
            pad_size = K_padded - K_orig
            # Pad K dimension (rows) of B_ternary_unpacked
            B_ternary_padded = torch.nn.functional.pad(B_ternary_unpacked, (0, 0, 0, pad_size), "constant", 0) 

        # 4. Pack padded ternary weights
        # B_ternary_padded is (K_padded, N), int8
        B_packed_bytes = pack_ternary_weights_to_int8(B_ternary_padded) # (K_padded/4, N), int8

        # 5. Prepare output tensor C
        C_out_quant = torch.empty((M, N), dtype=torch.float32, device=A_fp.device)

        # 6. Launch Triton Kernel
        # Ensure BLOCK_SIZE_K in autotuner configs are multiples of 4.
        # The kernel itself should also assert or handle this.
        # The K argument to the kernel is K_padded.
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        # Strides for B_packed_bytes (K_padded/4, N)
        stride_bpk, stride_bpn = B_packed_bytes.stride()

        bit_matmul_kernel[grid](
            A_q_padded, B_packed_bytes, C_out_quant,
            M, N, K_padded,
            A_q_padded.stride(0), A_q_padded.stride(1),
            stride_bpk, stride_bpn,
            C_out_quant.stride(0), C_out_quant.stride(1),
            # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K are from autotuner
        )
        # C_out_quant is now populated with int32 results cast to float32

        # 7. Dequantize output
        # Output Y = (A_q @ B_q) / (A_scale * B_scale_for_ternary_weights)
        # Note: B_ternary_scale is 1/mean(abs(W)). So W_approx = B_ternary_unpacked / B_ternary_scale.
        # A_approx = A_q / A_scale
        # So, Y_approx = (A_q B_ternary_unpacked) / (A_scale * B_ternary_scale) is not quite right.
        # It should be Y_approx = (A_q B_ternary_unpacked) * (1/A_scale) * (1/B_ternary_scale)
        # where (1/A_scale) is scale_A_inv and (1/B_ternary_scale) is scale_B_inv (which is mean(abs(W)))
        
        # Let's clarify scaling:
        # A_fp ≈ A_q / A_scale
        # B_fp ≈ B_ternary_unpacked / B_ternary_scale (where B_ternary_scale = 1 / mean(abs(B_fp)))
        # So, A_fp @ B_fp ≈ (A_q @ B_ternary_unpacked) / (A_scale * B_ternary_scale)
        
        dequant_factor = A_scale * B_ternary_scale
        if abs(dequant_factor) < 1e-9 : # Avoid division by zero if scales are tiny
            C_out_dequant = C_out_quant * 0.0 # effectively zero out if scales are bad
        else:
            C_out_dequant = C_out_quant / dequant_factor

        # Save for backward: unpadded quantized A, unpadded ternary B, and their scales.
        # Gradients are w.r.t. A_fp and B_fp.
        ctx.save_for_backward(A_q, B_ternary_unpacked, A_scale, B_ternary_scale)
        
        return C_out_dequant

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is dL/dC_out_dequant
        A_q, B_ternary_unpacked, A_scale, B_ternary_scale = ctx.saved_tensors
        
        grad_A_fp = None
        grad_B_fp = None

        # Using Straight-Through Estimator (STE) logic:
        # Y_dequant = (A_q @ B_ternary_unpacked) / (A_scale * B_ternary_scale)
        # A_approx = A_q / A_scale
        # B_approx = B_ternary_unpacked / B_ternary_scale
        # dL/dA_fp effectively becomes dL/dA_approx
        # dL/dB_fp effectively becomes dL/dB_approx
        
        # B_approx for gradient calculation needs to be float
        B_approx = B_ternary_unpacked.to(grad_output.dtype) / B_ternary_scale
        
        # A_approx for gradient calculation needs to be float
        A_approx = A_q.to(grad_output.dtype) / A_scale

        if ctx.needs_input_grad[0]: # Gradient for A_fp
            grad_A_fp = grad_output @ B_approx.T
        
        if ctx.needs_input_grad[1]: # Gradient for B_fp
            grad_B_fp = A_approx.T @ grad_output
            
        return grad_A_fp, grad_B_fp, None # None for K_original

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, use_rms_norm=True, eps=1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        # No bias for BitLinear as per common implementations, can be added if needed
        # self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            self.rms_norm = RMSNorm(in_features, eps=eps)

    def reset_parameters(self):
        # Standard kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if hasattr(self, 'bias') and self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x shape: (batch_size, ..., in_features)
        
        if self.use_rms_norm:
            x_normed = self.rms_norm(x)
        else:
            x_normed = x
            
        # The BitMatmulFunction expects (M, K) for A, (K, N) for B.
        # If x is (batch, seq_len, in_features), we might need to reshape.
        # For a simple MLP, x is likely (batch_size, in_features).
        original_shape = x_normed.shape
        if len(original_shape) > 2:
            M = original_shape[:-1].numel() # Product of all dims except last
            x_reshaped = x_normed.reshape(M, self.in_features)
        else:
            x_reshaped = x_normed # Should be (batch_size, in_features)

        # Pass K_original (self.in_features) to the function for padding logic
        output = BitMatmulFunction.apply(x_reshaped, self.weight, self.in_features)
        
        if len(original_shape) > 2:
            output = output.reshape(*original_shape[:-1], self.out_features)
            
        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, use_rms_norm={self.use_rms_norm}'
