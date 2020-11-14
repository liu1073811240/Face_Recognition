import torch

a = torch.tensor([1, 2, 3, 4, 5])

# 取布尔值
print(a < 4)  # tensor([ True,  True,  True, False, False])
print(torch.lt(a, 4))  # tensor([ True,  True,  True, False, False])
# lt gt eq le ge

# 取值
print(a[a < 4])  # tensor([1, 2, 3])
print(torch.masked_select(a, a < 4))  # tensor([1, 2, 3])

# 取索引，默认是非0元素的索引
print(torch.nonzero(a < 4, as_tuple=False))
# tensor([[0],
#         [1],
#         [2]])






