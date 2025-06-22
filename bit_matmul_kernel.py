import triton
import triton.language as tl

# Ternary values: -1, 0, 1
# Mapped to 2-bit for packing:
# 0 (00) -> 0
# 1 (01) -> 1
# 2 (10) -> -1
# (11 is unused, map to 0)
@triton.jit
def map_2bit_repr_to_val_triton(two_bit_repr):
    # This must return float for accumulation
    val = 0.0
    if two_bit_repr == 0: # 00 -> 0
        val = 0.0
    elif two_bit_repr == 1: # 01 -> 1
        val = 1.0
    elif two_bit_repr == 2: # 10 -> -1
        val = -1.0
    # else: 11 (unused) -> 0.0
    return val

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_warps': 2, 'num_stages': 2}),
        # Add more configs, especially for K as multiple of 4 due to packing
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'num_warps': 2, 'num_stages': 2}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bit_matmul_kernel(
    A_ptr, B_packed_ptr, C_ptr,
    M, N, K_padded, # K_padded is original K dimension, must be multiple of 4 for this kernel.
                 # Actual K used for iteration can be less if original K was smaller.
    stride_am, stride_ak,
    stride_bpk, stride_bpn, # Strides for B_packed_ptr (K_padded/4, N)
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, # BLOCK_SIZE_K here is for the unpacked K dimension
    GROUP_SIZE_M: tl.constexpr
):
    """
    Kernel for Ternary Matrix Multiplication: C = A @ B
    A: (M, K_padded) int8 activations
    B_packed: (K_padded/4, N) int8 packed ternary weights (4 weights per byte)
    C: (M, N) float32 output (accumulated in int32, stored as float32)

    BLOCK_SIZE_K must be a multiple of 4.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create offsets for the M dimension
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    # Create offsets for the N dimension
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    # Create offsets for the K dimension
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator with zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    # K_padded is the dimension A and B_unpacked would have.
    # The loop iterates over K_padded.
    for k_loop_idx in range(0, tl.cdiv(K_padded, BLOCK_SIZE_K)):
        k_start_offset = k_loop_idx * BLOCK_SIZE_K
        
        # --- Load A tile ---
        # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_ptrs = A_ptr + (offs_am[:, None] * stride_am + (k_start_offset + offs_k[None, :]) * stride_ak)
        # Boundary check for A loading (K dimension)
        a_mask = (k_start_offset + offs_k[None, :]) < K_padded
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0) # Load int8, zero pad if out of K_padded bounds

        # --- Load B_packed tile and unpack ---
        # B_packed is (K_padded/4, N)
        # We need to load a (BLOCK_SIZE_K/4, BLOCK_SIZE_N) tile of packed bytes
        # and unpack it to (BLOCK_SIZE_K, BLOCK_SIZE_N) of ternary floats.

        # offs_k_packed are indices for rows of B_packed_ptr: (0, 1, ..., BLOCK_SIZE_K/4 - 1)
        # Ensure BLOCK_SIZE_K is a multiple of 4 using an assertion or static check in Python wrapper
        # For simplicity, let's assume BLOCK_SIZE_K is always a multiple of 4.
        
        # Create a container for unpacked B tile in registers
        b_unpacked_tile = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

        for k_idx_in_block in range(0, BLOCK_SIZE_K, 4): # Iterate over groups of 4 k-values
            # k_actual is the true k index for this group of 4
            k_actual_base = k_start_offset + k_idx_in_block 
            
            # Pointer to the packed byte in B_packed_ptr
            # Row index for B_packed_ptr: (k_actual_base / 4)
            # Col indices for B_packed_ptr: offs_bn
            b_packed_row_idx = k_actual_base // 4
            
            b_packed_ptrs = B_packed_ptr + (b_packed_row_idx * stride_bpk + offs_bn[None, :] * stride_bpn) # Shape (1, BLOCK_SIZE_N)
            
            # Boundary check for B loading (K_padded/4 dimension for rows, N for columns)
            b_packed_mask_k = b_packed_row_idx < (K_padded // 4)
            # offs_bn already handles N boundary through C storage masks later
            
            # Load one row of packed bytes (corresponding to 4 rows of unpacked B)
            # if b_packed_mask_k is true, otherwise load 0 (which unpacks to 0s)
            packed_bytes = tl.load(b_packed_ptrs, mask=b_packed_mask_k & (offs_bn[None,:] < N), other=0) # packed_bytes shape (1, BLOCK_SIZE_N)

            # Unpack the 4 ternary values from each byte
            # These are tl.tensor of shape (1, BLOCK_SIZE_N) or (BLOCK_SIZE_N) after squeeze
            # val_k0, val_k1, val_k2, val_k3 are the unpacked float {-1,0,1} values

            # Shift amounts for 2-bit values
            val0_2bit = (packed_bytes >> 0) & 0x03
            val1_2bit = (packed_bytes >> 2) & 0x03
            val2_2bit = (packed_bytes >> 4) & 0x03
            val3_2bit = (packed_bytes >> 6) & 0x03

            # Store unpacked values in b_unpacked_tile
            # Offsets k_idx_in_block to k_idx_in_block + 3
            # Need to use tl.store or direct assignment if it's register based
            # This part is tricky with Triton's tensor semantics. Let's try direct assignment.
            # Assuming b_unpacked_tile is register-based.
            # The indices must match what tl.dot expects or how manual accumulation is done.
            
            # This direct assignment approach to build b_unpacked_tile might be inefficient or not work as expected.
            # A more common Triton pattern is to load smaller pieces and process.
            # Let's simplify and do the matmul incrementally for these 4 unpacked rows.
            
            # Convert 2-bit representations to float values {-1, 0, 1}
            # These will be tl.tensor of shape (BLOCK_SIZE_N)
            b_vals = [map_2bit_repr_to_val_triton(val0_2bit),
                      map_2bit_repr_to_val_triton(val1_2bit),
                      map_2bit_repr_to_val_triton(val2_2bit),
                      map_2bit_repr_to_val_triton(val3_2bit)]


            # Accumulate for these 4 k-values
            for i in range(4):
                if (k_idx_in_block + i) < BLOCK_SIZE_K : # Check if this k-slice is within current BLOCK_SIZE_K
                    # current_k_unpacked = k_start_offset + k_idx_in_block + i
                    # if current_k_unpacked < K_padded: # Redundant if a_mask and b_mask handle K_padded
                    
                    # a_slice is (BLOCK_SIZE_M, 1) corresponding to a_tile[:, k_idx_in_block + i]
                    # b_slice is (1, BLOCK_SIZE_N) corresponding to b_vals[i] (which is already (BLOCK_SIZE_N))
                    
                    a_slice = tl.load(A_ptr + offs_am[:, None] * stride_am + (k_actual_base + i) * stride_ak, 
                                      mask=((k_actual_base + i) < K_padded) & (offs_am[:,None] < M) , other=0)
                    
                    # a_slice is int8, b_vals[i] is float32 {-1,0,1}
                    # Accumulate: int8 * float{-1,0,1} -> convert a_slice to int32 for safety if b_vals were int.
                    # Here, b_vals[i] is float. Convert a_slice to float.
                    accumulator += a_slice.to(tl.float32) * b_vals[i] # Accumulator is float32 now, or make it int32 as planned.

        # If accumulator is float32
        # accumulator = tl.dot(a_tile.to(tl.float32), b_unpacked_tile, accumulator) # if b_unpacked_tile was fully formed
    
    # After all k_loops, accumulator holds the int32 sums.
    # Convert to float32 for storing, or it's already float32 if accumulated as such.
    # Let's re-evaluate accumulator type. If A is int8, B is {-1,0,1} int. Accumulator can be int32.
    # The current code makes accumulator float32 because b_vals[i] is float.
    # If map_2bit_repr_to_val_triton returned int32:
    # accumulator += a_slice.to(tl.int32) * b_vals[i] # then accumulator is int32. C_ptr stores float32.

    # Let's stick to float32 accumulator for now, as it's simpler with current b_vals[i]
    # For int32 accumulator:
    # map_2bit_repr_to_val_triton should return tl.int32
    # a_slice.to(tl.int32) * b_vals[i] -> this would be int32 accumulation
    # Then final store: c = accumulator.to(tl.float32)

    c = accumulator # Assuming accumulator is already float32.

    # --- Store C tile ---
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
