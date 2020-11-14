a = [1, 2, 3]

a.append(4)
print(a)  # [1, 2, 3, 4]

b = [5, 6]

# 展平成 当前维的数据
a.extend(b)
print(a)  # [1, 2, 3, 4, 5, 6]

a.append(b)
print(a)  # [1, 2, 3, 4, 5, 6, [5, 6]]
