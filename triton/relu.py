import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask= mask)
    tl.store( out_ptr + offsets, tl.maximum(x, 0.0), mask = mask)

def relu(x):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    add_kernel[grid](x, out, n, BLOCK_SIZE = 1024)
    return out

x = torch.rand(1_000_000, device = 'cuda')
out = relu(x)
print( "max error = ", (out - x).abs().max().item())