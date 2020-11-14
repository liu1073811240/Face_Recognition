import numpy as np


def iou(box,boxes,isMin = False):

    box_area = (box[2]-box[0])*(box[3]-box[1])  # [x1,y1,x2,y2,c]
    boxes_area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2-xx1)
    h = np.maximum(0, yy2-yy1)

    inter = w*h
    if isMin:
        #最小面积
        over = np.true_divide(inter, np.minimum(box_area, boxes_area))
    else:
        # 并集
        over = np.true_divide(inter, (box_area+boxes_area-inter))
    return over


def nms(boxes, thresh=0.3, isMin=False):
    if boxes.shape[0] == 0:
        return np.array([])
    # 根据置信度排序
    _boxes = boxes[(-boxes[:, 4]).argsort()]
    # 保留剩余的框
    r_boxes = []
    while _boxes.shape[0] > 1:
        # 取出第一个框
        a_box = _boxes[0]
        b_box = _boxes[1:]
        # 保留第一个框
        r_boxes.append(a_box)

        # 比较iou后保留阈值小的值
        index = np.where(iou(a_box, b_box, isMin) < thresh)
        _boxes = b_box[index]
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])
    return np.stack(r_boxes)


def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3]-bbox[:, 1]
    w = bbox[:, 2]-bbox[:, 0]
    max_side = np.maximum(w, h)
    square_bbox[:, 0] = bbox[:, 0]+w*0.5-max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0]+max_side
    square_bbox[:, 3] = square_bbox[:, 1]+max_side

    return square_bbox


if __name__ == '__main__':

    # bs = np.array([[2, 2, 30, 30, 40], [3, 3, 25, 25, 60], [18, 18, 27, 27, 15]])
    # print(nms(bs))

    box = np.array([2, 2, 30, 30, 40])
    boxes = np.array([[2, 2, 30, 30, 40], [3, 3, 25, 25, 60], [18, 18, 27, 27, 15]])
    iou(box, boxes, isMin=False)