# -*- coding: iso-8859-1 -*-
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import random

model = YOLO("C:/ImageDetection/ImageLabeling2/yolov8m_custom2.pt")
video_path = "C:/ImageDetection/Segmentation1/Frango/hiv00001.mp4"
cap = cv2.VideoCapture(video_path)
centroids = {}  
next_id = 0  
dist = 15

vibrant_colors = [(255, 0, 0),   
                  (0, 255, 0),  
                  (0, 0, 255),  
                  (255, 255, 0),
                  (255, 0, 255), 
                  (0, 255, 255)] 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    
    for r in results:
        for b in r.boxes:
            x_center = (b.xyxy[0][0] + b.xyxy[0][2]) / 2
            y_center = (b.xyxy[0][1] + b.xyxy[0][3]) / 2
            
            found_match = False
            for obj_id, (coordinates_list, color) in centroids.items(): 
                cv2.rectangle(frame, (int(b.xyxy[0][0]), int(b.xyxy[0][3])), (int(b.xyxy[0][2]), int(b.xyxy[0][1])), color, 2)
                last_x, last_y = coordinates_list[-1] 
                if abs(last_x - int(x_center)) < dist and abs(last_y - int(y_center)) < dist:
                    coordinates_list.append((int(x_center), int(y_center)))
                    found_match = True
                    break
            if not found_match:
                new_id = next_id
                next_id += 1
                color = random.choice(vibrant_colors)
                centroids[new_id] = ([(int(x_center), int(y_center))], color)

    for obj_id, (coordinates_list, color) in list(centroids.items()):
        if len(coordinates_list) > 1:
            for i in range(len(coordinates_list) - 1):
                cv2.line(frame, coordinates_list[i], coordinates_list[i + 1], color, 2)

    cv2.imshow("Detections", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


