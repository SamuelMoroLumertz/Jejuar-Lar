#<<<<<<< HEAD
## -*- coding: iso-8859-1 -*-


###### CÓDIGO USANDO TRACKING DO YOLOV8
#=======
## -*- coding: utf-8 -*-
## CÃ“DIGO USANDO TRACKING DO YOLOV8
#>>>>>>> 856c337ad7f6c8be0e083b0905c6498532600835

#import cv2
#from ultralytics import YOLO
#from collections import defaultdict
#import numpy as np
#import os
#import time
#from Frango import Frango

#<<<<<<< HEAD
#class Galinha:
#    def __init__(self, galinha_id, pos_x, pos_y):
#        self.galinha_id = galinha_id
#        self.distancia_percorrida = 0
#        self.tempo_parado = 0
#        self.pos_x = pos_x
#        self.pos_y = pos_y

#    def atualizar_posicao(self, new_pos_x, new_pos_y):
#        # Calcula a distância percorrida desde a última atualização
#        distancia = ((new_pos_x - self.pos_x) ** 2 + (new_pos_y - self.pos_y) ** 2) ** 0.5
#        self.distancia_percorrida += distancia

#        # Atualiza a posição atual
#        self.pos_x = new_pos_x
#        self.pos_y = new_pos_y

#    def incrementar_tempo_parado(self):
#        self.tempo_parado += 1

#model = YOLO("C:/ImageDetection/ImageLabeling2/yolov8m_custom2.pt")
#video_path = "C:/ImageDetection/Segmentation1/Frango/hiv00001.mp4"
#=======
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#model = YOLO("yolov8m_custom2.pt")
#video_path = "../../Video/hiv00001.mp4"
#>>>>>>> 856c337ad7f6c8be0e083b0905c6498532600835
#cap = cv2.VideoCapture(video_path)

##Store the track history
## track_history = defaultdict(lambda: [])
##
## while True:
##     ret, frame = cap.read()
##     if not ret:
##         break
##     results = model.track(frame, persist=True, show_labels=False, show_conf=False)
##     # Get the boxes and track IDs
##     boxes = results[0].boxes.xywh.cpu()
##     track_ids = results[0].boxes.id.int().cpu().tolist()
##
##     for box, track_id in zip(boxes, track_ids):
##         x, y, w, h = box
##
##         x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
##         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
##         t_size = cv2.getTextSize(str(track_id), 0, fontScale=0.5, thickness=1)[0]
##         cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1)+3),  (0, 255, 0), -1)
##         cv2.putText(frame, str(track_id), (int(x1), int(y1) - 2),
##                     0, 0.5, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
##
##         track = track_history[track_id]
##         track.append((float(x), float(y)))
##         if len(track) > 3:
##             track.pop(0)
##
##
##         points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
##         cv2.polylines(frame, [points], isClosed=False, color=(230, 0, 0), thickness=1)
##
##     cv2.namedWindow("Detections", cv2.WINDOW_AUTOSIZE)
##     cv2.imshow("Detections", frame)
##     if cv2.waitKey(1) & 0xFF == ord('q'):
##         break
##
## cap.release()
## cv2.destroyAllWindows()

#<<<<<<< HEAD
## Criar o dicionário para armazenar informações das galinhas
#galinhas_dict = {}

#=======

#frangos = []
#id_track = 0;
#frame_count = 0;
#limit = 4;
#>>>>>>> 856c337ad7f6c8be0e083b0905c6498532600835
#while True:
#    ret, frame = cap.read()
#    if not ret:
#        break
#    results = model.track(frame, persist=True, show_labels=False, show_conf=False)
#    # Get the boxes and track IDs
#    boxes = results[0].boxes.xywh.cpu()
#    track_ids = results[0].boxes.id.int().cpu().tolist()

#    for box in boxes:
#        x, y, w, h = box


#<<<<<<< HEAD
#        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2)) 
#        cv2.polylines(frame, [points], isClosed=False, color=(230, 0, 0), thickness=10)
#=======
#        x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
#>>>>>>> 856c337ad7f6c8be0e083b0905c6498532600835

#        inactive_boxes = []
#        inactive_id = []

#        for frango in frangos:
#            if frango.is_inactive(1):
#                inactive_id.append(frango.id)
#                inactive_boxes.append(frango)

#        for frango in inactive_boxes:
#            for ave in frangos:
#                if ave.is_close(frango.xmin, frango.ymin, frango.xmax, frango.ymax) and len(inactive_id) > 0:
#                    ave.id = inactive_id[0]
#                    inactive_id.pop(0)
#            frangos.remove(frango)




#        found_frango = False

#        for frango in frangos:
#            if frango.is_close(x1,y1,x2,y2):
#                frango.update_bounding_box(x1,y1,x2,y2)
#                found_frango = True
#                break

#        if not found_frango:
#            if len(frangos) < len(track_ids):
#                for frango in frangos:
#                    if frango.is_close(x1,y1,x2,y2):
#                        frango.update_bounding_box(x1,y1,x2,y2)
#                        break
#                id_track += 1
#                new_frango = Frango(x1,y1,x2,y2,id_track)
#                frangos.append(new_frango)
#            else:
#                for frango in frangos:
#                    if frango.is_close(x1,y1,x2,y2):
#                        frango.update_bounding_box(x1,y1,x2,y2)
#                        break


#    for frango in frangos:
#        x1, y1, x2, y2, id = frango.xmin, frango.ymin, frango.xmax, frango.ymax, frango.id
#        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
#        t_size = cv2.getTextSize(str(id), 0, fontScale=0.5, thickness=1)[0]
#        cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1)+3),  (0, 255, 0), -1)
#        cv2.putText(frame, str(id), (int(x1), int(y1) - 2),
#                    0, 0.5, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)


#    cv2.namedWindow("Detections", cv2.WINDOW_AUTOSIZE)
#    cv2.imshow("Detections", frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
#cv2.destroyAllWindows()



#<<<<<<< HEAD

####CLASSE DOS FRANGOS
##ID
##DISTPERCORRIDA
##TEMPOPARADO
##POSX
##POSY
##TEMPOCOMENDO
##TEMPOBEBENDO
#=======
#>>>>>>> 856c337ad7f6c8be0e083b0905c6498532600835
