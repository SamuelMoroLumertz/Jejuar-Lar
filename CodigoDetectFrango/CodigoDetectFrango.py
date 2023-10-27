# -*- coding: iso-8859-1 -*-

#CÓDIGO USANDO TRACKING MANUALMENTE

#import cv2
#import numpy as np
#from ultralytics import YOLO

## Inicialize o YOLO para o primeiro frame
#model = YOLO("C:/ImageDetection/ImageLabeling2/yolov8m_custom2.pt")
#video_path = "C:/ImageDetection/Segmentation1/Frango/hiv00001.mp4"
#cap = cv2.VideoCapture(video_path)
#ret, first_frame = cap.read()
#results = model(first_frame)
#tracked_objects = []

## Função para calcular a distância entre dois pontos
#def distance(point1, point2):
#    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

#while True:
#    ret, frame = cap.read()
#    if not ret:
#        break

#    if not tracked_objects:
#        # Se a lista de objetos rastreados estiver vazia, execute a detecção YOLO novamente
#        results = model(frame)
#        for r in results:
#            for b in r.boxes:
#                x_center = (b.xyxy[0][0] + b.xyxy[0][2]) / 2
#                y_center = (b.xyxy[0][1] + b.xyxy[0][3]) / 2
#                tracked_objects.append({"centroid": (int(x_center), int(y_center))})
#    else:
#        # Rastreamento dos objetos a partir dos centroids iniciais
#        for obj in tracked_objects:
#            min_distance = float('inf')
#            closest_point = None
#            for r in results:
#                for b in r.boxes:
#                    x_center = (b.xyxy[0][0] + b.xyxy[0][2]) / 2
#                    y_center = (b.xyxy[0][1] + b.xyxy[0][3]) / 2
#                    dist = distance(obj["centroid"], (int(x_center), int(y_center)))
#                    if dist < min_distance:
#                        min_distance = dist
#                        closest_point = (int(x_center), int(y_center))
#            obj["centroid"] = closest_point
#            # Desenhe um círculo para o rastreamento
#            cv2.circle(frame, closest_point, 5, (0, 0, 255), -1)

#    cv2.imshow("Tracking", frame)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
#cv2.destroyAllWindows()














#from pathlib import Path
#import cv2
#import numpy as np
#from ultralytics import YOLO
#import random
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#model = YOLO("C:/ImageDetection/ImageLabeling2/yolov8m_custom2.pt")
#video_path = "C:/ImageDetection/Segmentation1/Frango/hiv00001.mp4"
#cap = cv2.VideoCapture(video_path)
#centroids = {}  
#next_id = 0  
#dist = 15

#vibrant_colors = [(255, 0, 0),   
#                  (0, 255, 0),  
#                  (0, 0, 255),  
#                  (255, 255, 0),
#                  (255, 0, 255), 
#                  (0, 255, 255)] 

#while True:
#    ret, frame = cap.read()
#    if not ret:
#        break
#    results = model(frame)
    
#    for r in results:
#        for b in r.boxes:
#            x_center = (b.xyxy[0][0] + b.xyxy[0][2]) / 2
#            y_center = (b.xyxy[0][1] + b.xyxy[0][3]) / 2
            
#            found_match = False
#            for obj_id, (coordinates_list, color) in centroids.items(): 
#                cv2.rectangle(frame, (int(b.xyxy[0][0]), int(b.xyxy[0][3])), (int(b.xyxy[0][2]), int(b.xyxy[0][1])), color, 2)
#                last_x, last_y = coordinates_list[-1] 
#                if abs(last_x - int(x_center)) < dist and abs(last_y - int(y_center)) < dist:
#                    coordinates_list.append((int(x_center), int(y_center)))
#                    found_match = True
#                    break
#            if not found_match:
#                new_id = next_id
#                next_id += 1
#                color = random.choice(vibrant_colors)
#                centroids[new_id] = ([(int(x_center), int(y_center))], color)

#    for obj_id, (coordinates_list, color) in list(centroids.items()):
#        if len(coordinates_list) > 1:
#            for i in range(len(coordinates_list) - 1):
#                cv2.line(frame, coordinates_list[i], coordinates_list[i + 1], color, 2)

#    cv2.imshow("Detections", frame)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
#cv2.destroyAllWindows()




















##### CÓDIGO USANDO TRACKING DO YOLOV8

import cv2
from ultralytics import YOLO
from collections import defaultdict
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO("C:/ImageDetection/ImageLabeling2/yolov8m_custom2.pt")
video_path = "C:/ImageDetection/Segmentation1/Frango/hiv00001.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

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




#import cv2
#from ultralytics import YOLO
#from collections import defaultdict
#import os
#import numpy as np
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#model = YOLO("C:/ImageDetection/ImageLabeling2/yolov8m_custom2.pt")
#video_path = "C:/ImageDetection/Segmentation1/Frango/hiv00001.mp4"
#cap = cv2.VideoCapture(video_path)
#first=True
#bboxes = []

#tracker = cv2.legacy.TrackerCSRT_create()
#multiTracker = cv2.legacy.MultiTracker_create()

#while True:
#    ret, frame = cap.read()
#    if not ret:
#        break

#    if first:
#        results = model(frame)
#        boxes = results[0].boxes.xywh.cpu()

#        #ret = tracker.init(frame, (2,100,50,200))

#        for box in boxes:
#            x, y, w, h = box
#            x1, y1,x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
#            bbox = (int(x1),int(y1),int(w),int(h))
#            bboxes.append(bbox)
#            #multiTracker.add(tracker, frame, bbox)
#            #ret=tracker.init(frame, (int(x1),int(y1),int(w),int(h)))
#            #cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0, 0, 255), 1) 
            
#        first=False

#        for bbox in bboxes:
#            multiTracker.add(tracker, frame, bbox)

#        for i, newbox in enumerate(bboxes):
#            p1 = (int(newbox[0]), int(newbox[1]))
            
#            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
#            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
#    else:
#        # get updated location of objects in subsequent frames
#        success, boxes = multiTracker.update(frame)
        
#        # draw tracked objects
#        for i, newbox in enumerate(boxes):
#            p1 = (int(newbox[0]), int(newbox[1]))
            
#            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
#            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

#    cv2.imshow("Detections", frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
#cv2.destroyAllWindows()