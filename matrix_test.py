import torch
import time

# 检查CUDA是否可用
assert torch.cuda.is_available(), "CUDA不可用"
device = torch.device('cuda')

# 定义矩阵尺寸
n = 1000  # 可以调整矩阵大小以测试不同规模的计算
X = torch.randn(n, n, device=device)
A = torch.randn(n, 8, device=device)
B = torch.randn(8, n, device=device)

# 方法1: C=AB, Y=XC

C = torch.matmul(A, B)
start_time = time.time()
Y1 = torch.matmul(X, C)
method1_time = time.time() - start_time
print(Y1)

# 方法2: V=XA, Y=VB
start_time = time.time()
V = torch.matmul(X, A)
Y2 = torch.matmul(V, B)
method2_time = time.time() - start_time
print(Y2)

# 验证结果是否相同
# assert torch.allclose(Y1, Y2, atol=1e-6), "两种方法结果不一致"

# 打印运行时间
print(f"方法1 (C=AB, Y=XC) 运行时间: {method1_time:.6f} 秒")
print(f"方法2 (V=XA, Y=VB) 运行时间: {method2_time:.6f} 秒")
print(f"方法2比方法1快 {method1_time/method2_time:.2f} 倍")