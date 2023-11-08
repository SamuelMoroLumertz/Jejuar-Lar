# -*- coding: utf-8 -*-
# CÓDIGO USANDO TRACKING DO YOLOV8

import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import os
import time
from Frango import Frango

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO("yolov8m_custom2.pt")
video_path = "../../Video/hiv00001.mp4"
cap = cv2.VideoCapture(video_path)

# frangos = []
# id_track = 0;
# frame_count = 0;
# limit = 4;
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     results = model.track(frame, persist=True, show_labels=False, show_conf=False)
#     # Get the boxes and track IDs
#     boxes = results[0].boxes.xywh.cpu()
#     track_ids = results[0].boxes.id.int().cpu().tolist()
#
#     for box in boxes:
#         x, y, w, h = box
#
#
#         x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
#
#         inactive_boxes = []
#         inactive_id = []
#
#         for frango in frangos:
#             if frango.is_inactive(1):
#                 inactive_id.append(frango.id)
#                 inactive_boxes.append(frango)
#
#         for frango in inactive_boxes:
#             for ave in frangos:
#                 if ave.is_close(frango.xmin, frango.ymin, frango.xmax, frango.ymax) and len(inactive_id) > 0:
#                     ave.id = inactive_id[0]
#                     inactive_id.pop(0)
#             frangos.remove(frango)
#
#
#
#
#         found_frango = False
#
#         for frango in frangos:
#             if frango.is_close(x1,y1,x2,y2):
#                 frango.update_bounding_box(x1,y1,x2,y2)
#                 found_frango = True
#                 break
#
#         if not found_frango:
#             if len(frangos) < len(track_ids):
#                 for frango in frangos:
#                     if frango.is_close(x1,y1,x2,y2):
#                         frango.update_bounding_box(x1,y1,x2,y2)
#                         break
#                 id_track += 1
#                 new_frango = Frango(x1,y1,x2,y2,id_track)
#                 frangos.append(new_frango)
#             else:
#                 for frango in frangos:
#                     if frango.is_close(x1,y1,x2,y2):
#                         frango.update_bounding_box(x1,y1,x2,y2)
#                         break
#
#
#     for frango in frangos:
#         x1, y1, x2, y2, id = frango.xmin, frango.ymin, frango.xmax, frango.ymax, frango.id
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
#         t_size = cv2.getTextSize(str(id), 0, fontScale=0.5, thickness=1)[0]
#         cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1)+3),  (0, 255, 0), -1)
#         cv2.putText(frame, str(id), (int(x1), int(y1) - 2),
#                     0, 0.5, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
#
#
#     cv2.namedWindow("Detections", cv2.WINDOW_AUTOSIZE)
#     cv2.imshow("Detections", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
#
#
