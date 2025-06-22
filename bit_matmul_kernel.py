import triton
import triton.language as tl

@triton.jit
def map_2bit_repr_to_val_triton_int(two_bit_repr):
    val = 0 # int type
    if two_bit_repr == 0: # 00 -> 0
        val = 0
    elif two_bit_repr == 1: # 01 -> 1
        val = 1
    elif two_bit_repr == 2: # 10 -> -1
        val = -1
    # else: 11 (unused) -> 0
    return val

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'num_warps': 2, 'num_stages': 2}),
        # Add more, ensure BLOCK_SIZE_K is multiple of 4
    ],
    key=['M', 'N', 'K_padded'],
)
@triton.jit
def bit_matmul_kernel(
    A_ptr, B_packed_ptr, C_ptr,
    M, N, K_padded, # K_padded is original K dimension, must be multiple of 4.
    stride_am, stride_ak,
    stride_bpk, stride_bpn, # Strides for B_packed_ptr (K_padded/4, N)
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am_block = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn_block = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    # K_padded is the total K dimension to iterate over for the padded A and B.
    # BLOCK_SIZE_K must be a multiple of 4.
    if BLOCK_SIZE_K % 4 != 0:
        # This should ideally be a static assert or handled by autotuner config selection
        pass

    for k_outer_loop_idx in range(0, K_padded, BLOCK_SIZE_K):
        # --- Load A tile --- (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # offs_k_block are offsets within this BLOCK_SIZE_K part: [0, 1, ..., BLOCK_SIZE_K-1]
        offs_k_block = tl.arange(0, BLOCK_SIZE_K)
        
        a_ptrs = A_ptr + (offs_am_block[:, None] * stride_am + \
                          (k_outer_loop_idx + offs_k_block[None, :]) * stride_ak)
        
        # Mask for A loading based on K_padded and M
        a_mask = (offs_am_block[:, None] < M) & \
                   ((k_outer_loop_idx + offs_k_block[None, :]) < K_padded)
        a_tile_int8 = tl.load(a_ptrs, mask=a_mask, other=0) # int8
        a_tile = a_tile_int8.to(tl.int32) # Convert A tile to int32 for multiplication

        # --- Load B_packed tile and unpack in stages for dot product ---
        # B_packed is (K_padded/4, N)
        # We process 4 k-rows of B (unpacked) at a time, corresponding to 1 k-row of B_packed
        for k_inner_loop_idx in range(0, BLOCK_SIZE_K, 4): # k_inner_loop_idx steps by 4
            # Current base k index in the full K_padded dimension
            current_k_base = k_outer_loop_idx + k_inner_loop_idx

            # Row index for B_packed_ptr: (current_k_base / 4)
            b_packed_row_idx = current_k_base // 4
            
            # Ptrs to load a row of packed bytes from B_packed
            # Shape of loaded packed_bytes_row will be (BLOCK_SIZE_N)
            b_packed_ptrs = B_packed_ptr + (b_packed_row_idx * stride_bpk + \
                                            offs_bn_block[None, :] * stride_bpn) # Pointer to (1, BLOCK_SIZE_N) like data
            
            # Mask for loading packed_bytes_row
            # Check if this packed row index is within K_padded/4 and N
            b_load_mask = (b_packed_row_idx < (K_padded // 4)) & (offs_bn_block[None,:] < N)
            
            packed_bytes_row = tl.load(b_packed_ptrs, mask=b_load_mask, other=0).to(tl.int32) # Load as int32 for bitwise ops

            # Unpack the 4 ternary values for each element in packed_bytes_row
            # These will be tl.tensor of shape (BLOCK_SIZE_N)
            b_val_k0_repr = (packed_bytes_row >> 0) & 0x03
            b_val_k1_repr = (packed_bytes_row >> 2) & 0x03
            b_val_k2_repr = (packed_bytes_row >> 4) & 0x03
            b_val_k3_repr = (packed_bytes_row >> 6) & 0x03

            b_slice_0 = map_2bit_repr_to_val_triton_int(b_val_k0_repr) # shape (BLOCK_SIZE_N)
            b_slice_1 = map_2bit_repr_to_val_triton_int(b_val_k1_repr)
            b_slice_2 = map_2bit_repr_to_val_triton_int(b_val_k2_repr)
            b_slice_3 = map_2bit_repr_to_val_triton_int(b_val_k3_repr)

            # Corresponding A slices (columns from a_tile)
            # a_tile is (BLOCK_SIZE_M, BLOCK_SIZE_K)
            # k_inner_loop_idx is the offset within a_tile's K dimension
            a_slice_0 = tl.view(a_tile[:, k_inner_loop_idx + 0], (BLOCK_SIZE_M, 1))
            a_slice_1 = tl.view(a_tile[:, k_inner_loop_idx + 1], (BLOCK_SIZE_M, 1))
            a_slice_2 = tl.view(a_tile[:, k_inner_loop_idx + 2], (BLOCK_SIZE_M, 1))
            a_slice_3 = tl.view(a_tile[:, k_inner_loop_idx + 3], (BLOCK_SIZE_M, 1))
            
            # Perform dot products and accumulate
            # (BLOCK_SIZE_M,1) @ (1,BLOCK_SIZE_N) where b_slice is broadcasted from (BLOCK_SIZE_N)
            accumulator += a_slice_0 * b_slice_0[None, :]
            accumulator += a_slice_1 * b_slice_1[None, :]
            accumulator += a_slice_2 * b_slice_2[None, :]
            accumulator += a_slice_3 * b_slice_3[None, :]

    # Convert final int32 accumulator to float32 for storing
    c_float32 = accumulator.to(tl.float32)

    # --- Store C tile ---
    c_ptrs = C_ptr + offs_am_block[:, None] * stride_cm + offs_bn_block[None, :] * stride_cn
    c_mask = (offs_am_block[:, None] < M) & (offs_bn_block[None, :] < N)
    tl.store(c_ptrs, c_float32, mask=c_mask)
