import numpy as np
import imutils
import cv2

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (imageB, imageA) = images
        # Detectar pontos de interesse e extrair descritores
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # Fazer o match dos pontos entre as duas imagens
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # Se não houver matches, então não há alinhamento
        if M is None:
            return None

        # Caso contrário, aplicar uma perspectiva para alinhar as imagens
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # Verificar se os matches devem ser visualizados
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

            # Retorna uma tupla com a imagem alinhada e os matches
            return (result, vis)

        # Retorna a imagem alinhada
        return result

    def detectAndDescribe(self, image):
        # Converter a imagem para tons de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        # Verificar se estamos usando OpenCV 3.X ou melhor
        if self.isv3:
            # Detectar e descrever pontos de interesse
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # Caso contrário, estamos usando OpenCV 2.4.X
        else:
            # Detectar pontos de interesse
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # Extrair descritores
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # Converter os pontos de interesse para um array NumPy
        kps = np.float32([kp.pt for kp in kps])

        # Retornar uma tupla de pontos de interesse e descritores
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # Computar os matches entre os dois conjuntos de descritores
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # Verificar se o match é válido
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # Computar a homografia
        if len(matches) > 4:
            # Construir duas listas de pontos: um para cada imagem
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # Computar a homografia entre as duas imagens
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # Retornar os matches junto com a homografia e status dos matches
            return (matches, H, status)

        # Caso contrário, não há matches suficientes
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # Inicializar a visualização da imagem de saída
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # Loop sobre os matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # Verificar se o match foi um sucesso
            if s == 1:
                # Desenhar o match na imagem de saída
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # Retornar a imagem de saída
        return vis
