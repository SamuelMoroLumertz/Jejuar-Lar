import numpy as np
import cv2
import glob
import os

#chessBoardSize = (6,6)
#frameSize = (1280,720)

#def capture_photo():
#    cap = cv2.VideoCapture(1) 
#    photo_counter = 0

#    while True:
#        ret, frame = cap.read() 
        
#        if ret:
#            cv2.imshow('Webcam', frame)

#        key = cv2.waitKey(1)

#        if key == ord('p'):  # Ao pressionar a letra 'p' do teclado, tira uma foto e a salva
#            photo_counter += 1
#            cv2.imwrite('imagens/chessboard{}.png'.format(photo_counter), frame)
#        elif key == ord('q'):  # Ao pressionar a letra 'q' do teclado, finaliza o processo
#            break

#    cap.release()  # Libera a webcam
#    cv2.destroyAllWindows()

#def calibration():
#    criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001

#    objp = np.zeros((chessBoardSize[0]*chessBoardSize[1],3), np.float32)
#    objp[:,:2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1,2)

    
#    objpoints = [] 
#    imgpoints = [] 

#    images = glob.glob('imagens/camera320WS/*.jpg')

#    for image in images:
        
#        img = cv2.imread(image)
#        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
#        ret, corners = cv2.findChessboardCorners(gray, chessBoardSize, None)
        
#        if ret == True:
#            objpoints.append(objp)

#            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#            imgpoints.append(corners2)

#            img = cv2.drawChessboardCorners(img, chessBoardSize, corners2, ret)
#            cv2.imshow('img',img)
#            cv2.waitKey(500)

#    cv2.destroyAllWindows()

#    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)


#    output_folder = 'imagens/camera320WS/Resultados'

#    if not os.path.exists(output_folder):
#        os.makedirs(output_folder)

#    for image_path in images:
#        img = cv2.imread(image_path)
#        h,  w = img.shape[:2]
#       # newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

#        newCameraMtx = mtx
#        dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)

#        # dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)
#        # x, y, w, h = roi
#        # dst = dst[y:y+h, x:x+w]

#        output_path = os.path.join(output_folder, os.path.basename(image_path))
#        cv2.imwrite(output_path, dst)

#    output_file = 'imagens/camera320WS/Resultados/calibration_results.txt'

#    with open(output_file, 'w') as f:
#        f.write("Camera matrix:\n")
#        np.savetxt(f, mtx, fmt='%.8f')
#        f.write("\nDistortion coefficients:\n")
#        np.savetxt(f, dist, fmt='%.8f')

#    # img = cv2.imread('imagens/cameraSteel/chessBoard.jpg')
#    # h,  w = img.shape[:2]
#    # newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#    #
#    # dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)
#    # x, y, w, h = roi
#    # dst = dst[y:y+h, x:x+w]
#    # cv2.imwrite('imagens/resultados/resultado.jpg', dst)

##capture_photo()
#calibration()



img = cv2.imread("C:/Users/samue/OneDrive/Imagens/Capturas de tela/Captura de tela 2023-11-16 132719.png")
mtx= [834.94812252,0.00000000,637.95198468,0.00000000,834.43383649,302.29273117,0.00000000,0.00000000,1.00000000]
dist=[-0.34945005,0.18133822,0.00095096,0.00142214,-0.12601488]
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imshow("saldkfjalksdjf", dst)