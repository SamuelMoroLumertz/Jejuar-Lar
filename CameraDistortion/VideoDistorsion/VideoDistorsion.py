import openCV as cv2
import os
import numpy as np

video_path = "C:/Users/samue/Downloads/20231103_185601_tp00004.mp4"
cap = cv2.VideoCapture(video_path)
mtx = [
    [834.94812252, 0.00000000, 637.95198468],
    [0.00000000, 834.43383649, 302.29273117],
    [0.00000000, 0.00000000, 1.00000000]
]
dist = [-0.34945005, 0.18133822, 0.00095096, 0.00142214, -0.12601488]

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    resized = cv2.resize(frame, (1280,720), interpolation=cv2.INTER_AREA)
    dst = cv2.undistort(resized, np.array(mtx), np.array(dist), None, np.array(mtx))



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()