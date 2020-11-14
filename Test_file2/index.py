
path = r"D:\CelebA\list_bbox_celeba.txt"
with open(path, "r", encoding='utf-8') as f:
    rows = f.readlines()
    print(len(rows))

    i = 0
    for row in rows:
        try:
            city = row.split(' ')[1]
        except:
            continue

