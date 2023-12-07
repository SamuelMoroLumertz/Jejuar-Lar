import cv2
import tkinter as tk

# RTSP URL of the video streams
cameraDireita = "rtsp://camlab14b:senhasegura@10.57.1.19:554/stream1"
cameraEsquerda = "rtsp://camlab14b:senhasegura@10.57.1.20:554/stream2"
cameraAtual = cameraEsquerda

janela = (960, 540)
video = cv2.VideoCapture(cameraAtual)

# Get screen size
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

while True:
    ret, frame = video.read()

    frame = cv2.resize(frame, janela)
    cv2.imshow("RTSP", frame)

    # Calculate the position to center the window
    x = int((screen_width - janela[0]) / 2)
    y = int((screen_height - janela[1]) / 2)

    # Move the window to the center of the screen
    cv2.moveWindow("RTSP", x, y)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('t'):
        if cameraAtual == cameraEsquerda:
            cameraAtual = cameraDireita
        else:
            cameraAtual = cameraEsquerda
        video.release()
        video = cv2.VideoCapture(cameraAtual)

video.release()
cv2.destroyAllWindows()