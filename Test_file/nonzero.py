import torch

a = torch.tensor([[1, 2], [3, 4], [5, 6]])

# 取出对于元素的布尔值
print(a > 3)
print(a[a > 3])

# 取出每个轴的索引，默认是非0元素的索引
print(torch.nonzero(a > 3, as_tuple=False))

b = torch.tensor([[0,1,0,5,6,8],[1,2,0,0,5,0],[1,1,5,0,0,5]],dtype=torch.float32)
print(b)

# 取出非零元素的索引
print(b.nonzero())

