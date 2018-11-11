import urllib.request as ur
import cv2
import numpy as np

face_cascades = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_default.xml')
BASE_URL = 'http://192.168.254.101:8080/shot.jpg'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('../recognizer/trainingData.yml')

font = cv2.FONT_HERSHEY_COMPLEX
# font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SMALL, 5, 1, 0, 4)
while True:
    imageResponse = ur.urlopen(BASE_URL)
    imageNumArray = np.array(bytearray(imageResponse.read()), dtype=np.uint8)
    imageFrame = cv2.imdecode(imageNumArray, -1)
    grayscale = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
    faces = face_cascades.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        print(x, y, w, h)
        color= 0, 0, 255
        stroke = 2
        endCoordinateX = x + w
        endCoordinateY = y + h
        cv2.rectangle(imageFrame, (x, y), (endCoordinateX, endCoordinateY), color, stroke)
        id, confidence = recognizer.predict(grayscale[y:y+h, x:x+w])
        if id == 1:
            cv2.putText(imageFrame, str('Christian'), (x, y), font, 1, color, stroke, cv2.LINE_AA)
            print('Christian')

    cv2.imshow('Frame', imageFrame)
    # cv2.imshow('Grayscale Frame', grayscale)

    if cv2.waitKeyEx(20) & 0xFF == ord('q'):
            break

imageFrame.release()
cv2.destroyAllWindows()


