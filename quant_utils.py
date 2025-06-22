import torch

# Ternary values: -1, 0, 1
# Mapped to 2-bit for packing:
# 0 (00) -> 0
# 1 (01) -> 1
# 2 (10) -> -1
# (11 is unused, map to 0)

def map_val_to_2bit_repr(val):
    if val == 0:
        return 0
    elif val == 1:
        return 1
    elif val == -1:
        return 2
    else: # Should not happen with correctly quantized ternary weights
        raise ValueError(f"Invalid ternary value for 2-bit mapping: {val}")

def map_2bit_repr_to_val(two_bit_repr):
    if two_bit_repr == 0:
        return 0.0
    elif two_bit_repr == 1:
        return 1.0
    elif two_bit_repr == 2:
        return -1.0
    else: # unused 11, map to 0
        return 0.0

def quantize_ternary_weights_absmean(fp_weights, eps=1e-5):
    """
    Quantizes full-precision weights to {-1, 0, +1} based on their absolute mean.
    Output: torch.Tensor with values {-1, 0, +1} (int8).
            scale: The computed scale (1 / mean(abs(W))).
    """
    abs_mean = torch.mean(torch.abs(fp_weights))
    scale = 1.0 / (abs_mean + eps) # Add epsilon to prevent division by zero
    quantized_weights = torch.round(fp_weights * scale).clamp_(-1, 1)
    return quantized_weights.to(torch.int8), scale

def quantize_activations_absmax(fp_activations, n_bits=8):
    """
    Quantizes floating-point activations to n_bits integers using absmax quantization.
    Output: torch.Tensor (int8), scale (127 / max(abs(A))).
    """
    q_max = 2**(n_bits - 1) - 1 # For signed, e.g., 127 for int8
    q_min = -2**(n_bits - 1)    # For signed, e.g., -128 for int8
    
    abs_max_val = torch.max(torch.abs(fp_activations))
    scale = q_max / (abs_max_val + 1e-5) # Add epsilon to prevent division by zero if abs_max_val is 0
    
    quantized_activations = torch.round(fp_activations * scale).clamp_(q_min, q_max)
    return quantized_activations.to(torch.int8), scale

def pack_ternary_weights_to_int8(ternary_weights_int8):
    """
    Packs ternary weights (int8, values -1, 0, 1) into int8 bytes.
    4 ternary weights are packed into one byte.
    The K dimension (rows of weight matrix KxN) is packed.
    Assumes K is a multiple of 4. If not, it should be padded beforehand.
    Input: ternary_weights_int8 (K, N), K must be multiple of 4.
    Output: packed_weights (K//4, N), dtype=torch.int8
    """
    K, N = ternary_weights_int8.shape
    if K % 4 != 0:
        raise ValueError(f"K dimension ({K}) must be a multiple of 4 for packing.")

    packed_ K_dim = K // 4
    packed_weights = torch.empty((packed_K_dim, N), dtype=torch.int8, device=ternary_weights_int8.device)

    for i in range(packed_K_dim):
        b0 = map_val_to_2bit_repr(ternary_weights_int8[i*4 + 0, :])
        b1 = map_val_to_2bit_repr(ternary_weights_int8[i*4 + 1, :])
        b2 = map_val_to_2bit_repr(ternary_weights_int8[i*4 + 2, :])
        b3 = map_val_to_2bit_repr(ternary_weights_int8[i*4 + 3, :])
        
        # Pack: b3 b2 b1 b0 (b3 most significant 2 bits)
        packed_byte = (b3 << 6) | (b2 << 4) | (b1 << 2) | b0
        packed_weights[i, :] = packed_byte.to(torch.int8)
        
    return packed_weights

# For debugging or CPU reference - not used by Triton kernel directly
def unpack_int8_to_ternary_weights(packed_weights, original_K):
    """
    Unpacks int8 packed weights back to ternary weights (float32, values -1, 0, 1).
    Input: packed_weights (K//4, N), original_K
    Output: unpacked_weights (original_K, N)
    """
    packed_K_dim, N = packed_weights.shape
    if original_K // 4 != packed_K_dim :
        raise ValueError("Original K dimension doesn't match packed K.")

    unpacked_weights = torch.empty((original_K, N), dtype=torch.float32, device=packed_weights.device)
    for i in range(packed_K_dim):
        packed_byte_col = packed_weights[i, :].to(torch.int32) # Work with int32 for bitwise ops if needed, or ensure positive for >>
        
        b0_repr = (packed_byte_col >> 0) & 0x03
        b1_repr = (packed_byte_col >> 2) & 0x03
        b2_repr = (packed_byte_col >> 4) & 0x03
        b3_repr = (packed_byte_col >> 6) & 0x03
        
        unpacked_weights[i*4 + 0, :] = torch.tensor([map_2bit_repr_to_val(b.item()) for b in b0_repr], device=packed_weights.device)
        unpacked_weights[i*4 + 1, :] = torch.tensor([map_2bit_repr_to_val(b.item()) for b in b1_repr], device=packed_weights.device)
        unpacked_weights[i*4 + 2, :] = torch.tensor([map_2bit_repr_to_val(b.item()) for b in b2_repr], device=packed_weights.device)
        unpacked_weights[i*4 + 3, :] = torch.tensor([map_2bit_repr_to_val(b.item()) for b in b3_repr], device=packed_weights.device)
        
    return unpacked_weights
