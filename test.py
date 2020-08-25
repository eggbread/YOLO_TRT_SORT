import os
import time
import cv2
import pycuda.autoinit
import tensorrt as trt
from Detection.utils.yolo_classes import get_cls_dict
from Detection.utils.yolo_with_plugins import TrtYOLO
from Detection.utils.visualization import BBoxVisualization
from sort import *
import numpy as np

class Vision(object):
    def __init__(self, device, frame_width, frame_height):
        class_num = 80
        self.cls_dict = get_cls_dict(class_num)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.vis = BBoxVisualization(self.cls_dict)
        model = "yolov4-tiny-416"
        self.detector = TrtYOLO(model, (self.frame_width, self.frame_height), class_num)
        self.tracker = Sort()
        self.mode = 1

    def detect(self):
        cap = cv2.VideoCapture("/home/episci/Videos/oxford.mp4")
        cv2.namedWindow("frame")
        cls_dict = get_cls_dict(80)
        model = "yolov4-tiny-416"
        yolo_dim = model.split('-')[-1]
        h = w = int(yolo_dim)
        if h % 32 != 0 or w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        trt_yolo = TrtYOLO(model, (h, w),80)
        vis = BBoxVisualization(cls_dict)
        fps = 0
        start = time.time()
        while True:
            ret, img = cap.read()
            if ret:
                boxes, confs, clss = trt_yolo.detect(img, 0.3)
                boxes, confs, clss, oid = self.tracker.update(np.hstack((boxes, np.reshape(confs,(-1,1)), np.reshape(clss,(-1,1)))))
                img = vis.draw_bboxes(img, boxes, confs, clss, self.mode, oid)
                end = time.time()
                print(1/(end-start))
                start = end
                # cv2.imshow("frame",img)
                # cv2.waitKey(1)

if __name__=='__main__':
    v = Vision(0,416,416)
    print("Start")
    v.detect()
