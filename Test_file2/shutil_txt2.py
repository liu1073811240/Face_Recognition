import shutil
import os
import traceback

count = 10001

def replace_():
    path_txt = r"D:\ACelebA\negative3\negative2.txt"
    # with open(path_txt, "r") as f:

    file_data = ""
    count = 10001
    f = open(path_txt, "r")
    for i, line in enumerate(f):

        # print(i)
        # print(line)

        strs1 = line.split()
        # print(strs1)
        strs2 = strs1[0]
        strs3 = strs2.split("/")
        strs4 = strs3[1]

        print(strs4)
        # print(str(count) + '.jpg')
        if strs4 == str(count) + '.jpg':
            del line
            # after_line = line.replace('negative', '')
            # after_line = after_line.replace('/', '')
            # after_line = after_line.replace('strs4', '')
            #
            # after_line = after_line.replace('.jpg', '')
            # after_line = after_line.replace(' 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0', '')

            file_data += ""

            count += 1
        else:
            file_data += line

    with open(path_txt, "w") as f:
        f.write(file_data)


replace_()




