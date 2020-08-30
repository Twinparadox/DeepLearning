import cv2
import time
import numpy as np
from image_process.multi_image import create_image, create_image_multiple, show_multi_images

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("origin_classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("data/test1.mp4")
vc2 = cv2.VideoCapture("data/test2.mp4")

vc_list = [cv2.VideoCapture("data/test11.mp4"), cv2.VideoCapture("data/test12.mp4"), cv2.VideoCapture("data/test13.mp4"),
           cv2.VideoCapture("data/test21.mp4"), cv2.VideoCapture("data/test22.mp4"), cv2.VideoCapture("data/test23.mp4"),
           cv2.VideoCapture("data/test31.mp4"), cv2.VideoCapture("data/test32.mp4"), cv2.VideoCapture("data/test33.mp4")]

#net = cv2.dnn.readNet(f"data/yolo-obj_final.weights", "data/yolo-obj.cfg")
model_list = []
for idx in range(9):
    #net = cv2.dnn.readNet(f"data/yolo-obj_final.weights", "data/yolo-obj.cfg")
    net = cv2.dnn.readNet(f"data/yolov4-tiny.weights", "data/yolov4-tiny.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1 / 255)
    print(np.array(model.getMemoryConsumption((1, 3, 320,320)))[1] / (1024*1024))
    model_list.append(model)

width = 480
height = 360
depth = 3

while cv2.waitKey(1) < 1:
    total_time = 0
    total_draw = 0
    frame_list = []
    idx = 0
    for vc in vc_list:
        (grabbed, frame) = vc.read()
        if not grabbed:
            exit()

        start = time.time()
        classes, scores, boxes = model_list[idx].detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()

        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end_drawing = time.time()

        total_time += end - start
        total_draw += end_drawing - start_drawing
        frame = cv2.resize(frame, dsize=(480, 360), interpolation=cv2.INTER_AREA)
        frame_list.append(frame)

        idx += 1

    fps_label = "FPS: %.2f (Drawing time of %.2fms)" % (1 / (total_time + total_draw), (total_draw) * 1000)

    # 화면에 표시할 이미지 만들기 ( 2 x 2 )
    dstimage = create_image_multiple(height, width, depth, 3, 3)

    for idx in range(9):
        show_multi_images(dstimage, frame_list[idx], height, width, depth, idx//3, idx%3)

    cv2.putText(dstimage, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("detections", dstimage)