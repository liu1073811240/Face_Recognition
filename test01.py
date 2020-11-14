# 1、先拼接
image_path1 = r"D:\CelebA_test\5.txt"
image_path2 = r"D:\CelebA_test\10.txt"

result_path = r"D:\CelebA_test\result.txt"

with open(image_path1, 'r') as fa: # 读取需要拼接的前面那个TXT
    with open(image_path2, 'r') as fb: # 读取需要拼接的后面那个TXT

        with open(result_path, 'w') as fc: # 写入新的TXT
            for line in fa:
                fc.write(line.strip('\r\n')) # 用于移除字符串头尾指定的字符
                fc.write(line.join(" "))
                fc.write(fb.readline())