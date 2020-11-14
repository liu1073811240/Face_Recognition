import numpy as np

a = np.linspace(-1, 1, 5)  # 这是一个数组，数值在-1到1之间的5个数
lists = [a for _ in range(4)]  # 打印结果是：列表中存了4个数组
b = np.vstack(lists)

print(b)
for x in np.nditer(b):
    print(x)
    exit()

# print(a)
# print(lists)
# print(b)
# print(b[0])
# print('/n')
# for _ in range(4):
#     c = np.linspace(-1, 1, 5)
#     print(c)



