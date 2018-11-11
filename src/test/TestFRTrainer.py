import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
dirPath = '../datasets'

def getImagesWithId(dirPath):
    imagePaths = [os.path.join(dirPath, f) for f in os.listdir(dirPath)]
    faces = []
    IDs = []

    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')
        faceNumpy = np.array(faceImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNumpy)
        print(ID)
        IDs.append(ID)
        cv2.imshow("Training", faceNumpy)
        cv2.waitKeyEx(10)
    return np.array(IDs), faces

Ids, faces = getImagesWithId(dirPath)
recognizer.train(faces, np.array(Ids))
recognizer.save('../recognizer/trainingData.yml')
cv2.destroyAllWindows()

