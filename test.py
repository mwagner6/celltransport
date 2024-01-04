import torch
import time

dim=8000

x=torch.randn(dim,dim)
y=torch.randn(dim,dim)
start_time = time.time()
z=torch.matmul(x,y)
elapsed_time = time.time() - start_time
print('CPU_time = ',elapsed_time)


x=torch.randn(dim,dim,device=torch.device("cuda:0"))
y=torch.randn(dim,dim,device=torch.device("cuda:0"))
start_time = time.time()
z=torch.matmul(x,y)
elapsed_time = time.time() - start_time
print('GPU_time = ',elapsed_time)