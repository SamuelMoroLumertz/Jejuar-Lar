import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip

video_path = "../../../Videos/Camera.mp4"
output_path = "../../../Videos/"

cap = cv2.VideoCapture(video_path)
video_path = os.path.join(output_path, 'Camera6A.mp4')

mtx = [
    [834.94812252, 0.00000000, 637.95198468],
    [0.00000000, 834.43383649, 302.29273117],
    [0.00000000, 0.00000000, 1.00000000]
]
dist = [-0.34945005, 0.18133822, 0.00095096, 0.00142214, -0.12601488]
counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    dst = cv2.undistort(frame, np.array(mtx), np.array(dist), None, np.array(mtx))

    height, width, layers = dst.shape

    if counter == 0:
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

    video.write(dst)
    counter += 1

    if counter % 1000 == 0:
        print(f"{counter} imagens gravadas.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


# output_path_clip = os.path.join(output_path, 'Camera6_clip.mp4')
# clip = VideoFileClip(video_path)
# clip.write_videofile(output_path_clip, bitrate="768k", fps=15)
# clip.close()
