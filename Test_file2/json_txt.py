# -*- coding:utf-8 _*-
"""
@author:xxx
@file: read_json.py
@time: 2019/06/05
"""
# 根据标注精灵标好导出的json文件生成txt
import json
import os
import glob
from tqdm import tqdm


def get_files(path, _ends=['*.json']):
    all_files = []
    for _end in _ends:
        files = glob.glob(os.path.join(path, _end))
        all_files.extend(files)
    file_num = len(all_files)
    return all_files, file_num  # 获取所有文件、及文件个数


def get_text_mark(file_path):
    with open(file_path, 'r', encoding='utf-8') as fid:
        result_dict = json.load(fid)
        file_name = result_dict['path']
        strs = file_name.split('\\')

        width_size = result_dict['size']['width']
        height_size = result_dict['size']['height']

        obj = result_dict['outputs']['object']
        all_text_mark = []
        for obj_item in obj:
            text = obj_item['name']
            try:
                coords = obj_item['polygon']
                try:
                    output_coord = [int(float(coords['x1'])), int(float(coords['y1'])), int(float(coords['x2']))
                        , int(float(coords['y2'])), int(float(coords['x3'])), int(float(coords['y3'])),
                                    int(float(coords['x4'])), int(float(coords['y4']))]
                except:
                    continue
            except:
                coords = obj_item['bndbox']
                try:
                    # output_coord = [int(float(coords['xmin'])), int(float(coords['ymin'])), int(float(coords['xmax']))
                    #     , int(float(coords['ymin'])), int(float(coords['xmax'])), int(float(coords['ymax'])),
                    #                 int(float(coords['xmin'])), int(float(coords['ymax']))]

                    # output_coord = [int(float(coords['xmin'])), int(float(coords['ymin'])), int(float(coords['xmax'])),
                    #                 int(float(coords['ymax']))]
                    output_coord = [int(float(coords['xmin'])), int(float(coords['ymin'])), int(float(coords['ymax'] -
                                    float(coords['ymin']))),
                                    int(float(coords['xmax'] - float(coords['xmin'])))]
                except:
                    continue

            output_text = ''

            for item in output_coord:
                output_text = output_text + str(item) + ' '
            # output_text += text
            # output_text += strs[-1]
            output_text = strs[-1] + ' ' + output_text

            all_text_mark.append(output_text)
        return all_text_mark


def write_to_txt(out_txt_path, one_file_all_mark):
    # windows
    # with open(os.path.join(out_txt_path, file.split('\\')
    #                                      [-1].split('.')[0] + '.txt'), 'a+', encoding='utf-8') as fid:
    #     for item in one_file_all_mark:
    #         fid.write(item + '\n')

    with open(out_txt_path, 'a+') as f1:
        for item in one_file_all_mark:
            f1.write(item + '\n')


if __name__ == "__main__":
    json_path = r'G:\绿光浏览器下载\数据\outputs'
    out_txt_path = r'G:\绿光浏览器下载\数据\img.txt'
    files, files_len = get_files(json_path)
    bar = tqdm(total=files_len)  # 显示加载文件的进度条。
    for file in files:
        bar.update(1)
        print(file)
        try:
            one_file_all_mark = get_text_mark(file)
        except:
            # print(file)
            continue
        write_to_txt(out_txt_path, one_file_all_mark)
    bar.close()


