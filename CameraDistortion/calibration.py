import numpy as np
import cv2
import glob
import os

chessBoardSize = (6,6)
frameSize = (640,360)

def capture_photo():
    cap = cv2.VideoCapture(1) 
    photo_counter = 0

    while True:
        ret, frame = cap.read() 
        
        if ret:
            cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1)

        if key == ord('p'):  # Ao pressionar a letra 'p' do teclado, tira uma foto e a salva
            photo_counter += 1
            cv2.imwrite('imagens/chessboard{}.png'.format(photo_counter), frame)
        elif key == ord('q'):  # Ao pressionar a letra 'q' do teclado, finaliza o processo
            break

    cap.release()  # Libera a webcam
    cv2.destroyAllWindows()

def calibration():
    criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001

    objp = np.zeros((chessBoardSize[0]*chessBoardSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1,2)

    
    objpoints = [] 
    imgpoints = [] 

    images = glob.glob('imagens/cameraSteel/*.jpg')

    for image in images:
        
        img = cv2.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, chessBoardSize, None)
        
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, chessBoardSize, corners2, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    
    print("Camera matrix: \n", mtx)
    print("Distortion coefficients: \n", dist)

    output_folder = 'imagens/resultados'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_path in images:
        img = cv2.imread(image_path)
        h,  w = img.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, dst)
    # img = cv2.imread('imagens/cameraSteel/chessBoard.jpg')
    # h,  w = img.shape[:2]
    # newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    #
    # dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # cv2.imwrite('imagens/resultados/resultado.jpg', dst)

#capture_photo()
calibration()

