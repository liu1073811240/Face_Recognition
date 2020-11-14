import numpy as np
import os
from PIL import Image
import traceback

negative_img = r"D:\ACelebA\negative2"
save_path = r"D:\CelebA_20w"
def gen_sample(crop_size):
    negative_image_dir = os.path.join(save_path, str(crop_size), "negative")

    if not os.path.exists(negative_image_dir):
        os.makedirs(negative_image_dir)

    negative_anno_filename = os.path.join(save_path, str(crop_size), "negative.txt")
    negative_count = 400000

    try:
        negative_anno_file = open(negative_anno_filename, "a")

        try:
            for file in os.listdir(negative_img):
                img = Image.open(os.path.join(negative_img, file))
                img = img.convert("RGB")
                img_w, img_h = img.size
                print(np.shape(img))

                for _ in range(50):
                    side_len = np.random.randint(20, min(img_w, img_h))
                    x_ = np.random.randint(0, img_w-side_len)
                    y_ = np.random.randint(0, img_h-side_len)
                    if x_ < 0 or y_ < 0 or (x_+side_len) > img_w or (y_+side_len) > img_h:
                        continue

                    crop_boxes = np.array([x_, y_, x_+side_len, y_+side_len])
                    negative_image = img.crop(crop_boxes)
                    resize_image = negative_image.resize((crop_size, crop_size), Image.ANTIALIAS)

                    negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(
                                    negative_count, 0))
                    negative_anno_file.flush()
                    resize_image.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    negative_count += 1

                img.close()
        except:
            traceback.print_exc()
    finally:
        negative_anno_file.close()


if __name__ == '__main__':
    gen_sample(48)
    # gen_sample(24)
    # gen_sample(12)



