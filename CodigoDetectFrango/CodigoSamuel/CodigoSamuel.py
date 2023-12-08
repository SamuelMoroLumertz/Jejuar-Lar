# -*- coding: iso-8859-1 -*-
from asyncio.windows_events import NULL
import cv2
from ultralytics import YOLO
from collections import defaultdict
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #para evitar problemas de compatibilidade entre os ambientes

model = YOLO("C:/ImageDetection/ImageLabeling2/yolov8m_custom2.pt") #carrega o modelo que sera usado para detectar as galinhas
video_path = "C:/Users/samue/Downloads/20231103_173453_tp00002.mp4"
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])# Store the track history
galinhas=[]
contadorFrame=0
distMax=20

#criacao de uma classe GALINHA
class Galinha:
    def __init__(self, galinha_id, pos_x, pos_y):
        self.galinha_id = galinha_id
        self.distancia_percorrida = 0
        self.tempo_parado = 0
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.tempo_comendo = 0
        self.tempo_bebendo = 0

    def atualizar_posicao(self, new_pos_x, new_pos_y):
        # Calcula a distancia percorrida desde a ultima atualizacao
        distancia = ((new_pos_x - self.pos_x) ** 2 + (new_pos_y - self.pos_y) ** 2) ** 0.5
        self.distancia_percorrida += distancia
        # Atualiza a posicao atual
        self.pos_x = new_pos_x
        self.pos_y = new_pos_y

    def incrementar_tempo_parado(self):
        self.tempo_parado += 1

def galinha_existe(galinhas, id_procurado):
    for galinha in galinhas:
        if galinha.galinha_id == id_procurado:
            return galinha
    return False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    contadorFrame=contadorFrame+1

    results = model.track(frame, persist=True, show_labels=False, show_conf=False)
    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        x1, y1,x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
        galinha = galinha_existe(galinhas, track_id)   

        if galinha is not False:
            distancia = ((x - galinha.pos_x) ** 2 + (y - galinha.pos_y) ** 2) ** 0.5
            if contadorFrame%30==0 and distancia >= distMax:
                galinha.atualizar_posicao(x,y)
                print(f'A galinha {galinha.galinha_id} se mexeu! Distancia: {galinha.distancia_percorrida}')
            elif contadorFrame%30==0 and distancia < distMax:
                galinha.incrementar_tempo_parado()
                print(f'A galinha {galinha.galinha_id} ficou parada! Tempo Parado: {galinha.tempo_parado}')
        else:
            novaGalinha = Galinha(track_id,x,y)
            galinhas.append(novaGalinha)

        cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0, 0, 255), 1)
        t_size = cv2.getTextSize(str(track_id), 0, fontScale=1 / 1, thickness=1)[0]
        cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3),(int(x1) + t_size[0], int(y1)+3),  (0,255,0),-1)
        cv2.putText(frame, str(track_id), (int(x1), int(y1) - 2), 0, 1 / 1,[0,0,0], thickness=2, lineType=cv2.LINE_AA)
        track = track_history[track_id]
        track.append((float(x), float(y))) 
        
        if len(track) > 30: 
            track.pop(0)

        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2)) 
        cv2.polylines(frame, [points], isClosed=False, color=(230, 0, 0), thickness=5)

    cv2.imshow("Detections", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

####CLASSE DOS FRANGOS
##ID
##DISTPERCORRIDA
##TEMPOPARADO
##POSX
##POSY
##TEMPOCOMENDO
##TEMPOBEBENDO









#img = cv2.imread("C:/Users/samue/OneDrive/Imagens/Capturas de tela/Captura de tela 2023-11-16 132719.png")
#resized = cv2.resize(img, (1280,720), interpolation=cv2.INTER_AREA)
#mtx = [
#    [834.94812252, 0.00000000, 637.95198468],
#    [0.00000000, 834.43383649, 302.29273117],
#    [0.00000000, 0.00000000, 1.00000000]
#]

#dist = [-0.34945005, 0.18133822, 0.00095096, 0.00142214, -0.12601488]

#dst = cv2.undistort(resized, np.array(mtx), np.array(dist), None, np.array(mtx))

#cv2.namedWindow("resized", cv2.WINDOW_NORMAL)
#cv2.imshow("resized", dst)
#cv2.namedWindow("normal", cv2.WINDOW_NORMAL)
#cv2.imshow("normal", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()