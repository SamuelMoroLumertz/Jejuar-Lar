# -*- coding: iso-8859-1 -*-


##### CÓDIGO USANDO TRACKING DO YOLOV8

import cv2
from ultralytics import YOLO
from collections import defaultdict
import os
import numpy as np
from Galinha import Galinha

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


model = YOLO("C:/ImageDetection/ImageLabeling2/yolov8m_custom2.pt")
video_path = "C:/ImageDetection/Segmentation1/Frango/hiv00001.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Criar o dicionário para armazenar informações das galinhas
galinhas_dict = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True, show_labels=False, show_conf=False)
    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
    
        x1, y1,x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
        cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0, 0, 255), 1)
        t_size = cv2.getTextSize(str(track_id), 0, fontScale=1 / 1, thickness=1)[0]
        cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3),(int(x1) + t_size[0], int(y1)+3),  (0,255,0),-1)
        cv2.putText(frame, str(track_id), (int(x1), int(y1) - 2), 0, 1 / 1,[0,0,0], thickness=2, lineType=cv2.LINE_AA)

        track = track_history[track_id]
        track.append((float(x), float(y)))  
        if len(track) > 30: 
            track.pop(0)

        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2)) 
        cv2.polylines(frame, [points], isClosed=False, color=(230, 0, 0), thickness=10)

    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




###CLASSE DOS FRANGOS
#ID
#DISTPERCORRIDA
#TEMPOPARADO
#POSX
#POSY
#TEMPOCOMENDO
#TEMPOBEBENDO