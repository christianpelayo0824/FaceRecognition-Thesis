import urllib.request as ur
import cv2
import numpy as np

face_cascades = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_default.xml')
BASE_URL = 'http://192.168.254.101:8080/shot.jpg'

uid = input("Enter UID: ")
imageNumber = 0
while True:

    imageResponse = ur.urlopen(BASE_URL)
    imageNumArray = np.array(bytearray(imageResponse.read()), dtype=np.uint8)
    imageFrame = cv2.imdecode(imageNumArray, -1)
    grayscale = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
    faces = face_cascades.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=5)

    for x, y, w,h in faces:
        print(x, y, w, h)
        color= 0, 0, 255
        stroke = 2

        imageNumber += 1
        endCoordinateX = x + w
        endCoordinateY = y + h

        cv2.imwrite('../datasets/Image.'+str(uid)+'.'+str(imageNumber)+'.jpg', grayscale[y:y+h, x:x+w])
        cv2.rectangle(imageFrame, (x, y), (endCoordinateX, endCoordinateY), color, stroke)
        cv2.waitKey(100)
    cv2.imshow('Frame', imageFrame)
    cv2.waitKey(1)
    if imageNumber > 50:
        break

cv2.destroyAllWindows()


