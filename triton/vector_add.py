import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE : tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x =tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    tl.store (out_ptr + offsets , x +y, mask = mask)

def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024), )
    add_kernel[grid](x, y, out, n, BLOCK_SIZE = 1024)
    return out

x = torch.rand(1_000_000, device = 'cuda' )
y = torch.rand(1_000_000, device = 'cuda')
out = add(x,y)
print('max error = ', (out-(x+y)).abs().max().item())