from sticher import Stitcher
import argparse
import imutils
import cv2

# RTSP URL of the video streams
cameraDireita = "rtsp://camlab14b:senhasegura@10.57.1.19:554/stream1"
cameraEsquerda = "rtsp://camlab14b:senhasegura@10.57.1.20:554/stream2"
janela = (960, 540)
videoEsquerda = cv2.VideoCapture(cameraEsquerda)
videoDireita = cv2.VideoCapture(cameraDireita)

while True:
    ret, frameEsquerda = videoEsquerda.read()
    ret, frameDireita = videoDireita.read()

    frameEsquerda = cv2.resize(frameEsquerda, janela)
    frameDireita = cv2.resize(frameDireita, janela)
    frameEsquerda = imutils.resize(frameEsquerda, width=400)
    frameDireita = imutils.resize(frameDireita, width=400)

    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([frameEsquerda, frameDireita], showMatches=True)

    cv2.imshow("Frame", result)
    cv2.imshow("Vis", vis)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break