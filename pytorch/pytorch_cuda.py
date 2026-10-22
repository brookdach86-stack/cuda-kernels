import torch
import time

#检查GPU
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0) 
    print(f'GPU名称: {gpu.name}')
    print(f"SM 数量：{gpu.multi_processor_count}")
    print(f'显存：{gpu.total_memory/1024**3:.1f} GB') #转换成GB
    print(f'计算能力：{gpu.compute_capability}')

#GPU VS CPU 矩阵乘法性能对比
N = 4096
A_cpu = torch.randn(N, N)
B_cpu = torch.randn(N, N)
A_gpu = A_cpu.cuda()
B_gpu = B_cpu.cuda()

#预热
torch.mm(A_gpu, B_gpu) #matrix multiplication
torch.cuda.synchronize()
print("预热完成")

#计时
start = time.perf_counter() #高精度计时器
C_cpu = torch.mm(A_cpu, B_cpu) 
cpu_time = time.perf_counter() - start

torch.cuda.synchronize()
start = time.perf_counter()
C_gpu = torch.mm(A_gpu, B_gpu)
torch.cuda.synchronize()
gpu_time = time.perf_counter() - start

#转换成毫秒进行对比
print(f"CPU: {cpu_time*1000:.2f} ms | GPU: {gpu_time*1000:.2f} ms | 加速比: {cpu_time/gpu_time:.1f}x")


