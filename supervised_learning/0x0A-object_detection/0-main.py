#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('0-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    # size 13 116, 90 ,156 ...
    # size 26    30, 6q, 62 ....
    # size 52    1, 13, ....

    yolo_file = '../data/yolo.h5'
    classes_file = '../data/coco_classes.txt'
    yolo = Yolo(yolo_file, classes_file, 0.6, 0.5, anchors)
    yolo.model.summary()
    print('Class names:', yolo.class_names)
    print('Class threshold:', yolo.class_t)
    print('NMS threshold:', yolo.nms_t)
    print('Anchor boxes:', yolo.anchors)
