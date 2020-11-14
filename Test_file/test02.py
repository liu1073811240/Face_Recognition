
import torch
import torch.nn.functional as F

a = torch.randn([1, 512])
b = torch.matmul(F.normalize(a), F.normalize(a).t())
c = torch.dot(F.normalize(a).reshape(-1), F.normalize(a).reshape(-1))

print(b)  # tensor([[1.0000]])
print(c)

