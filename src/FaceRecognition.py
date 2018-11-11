import numpy as np
import cv2

face_cascades = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# face_cascades = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascades.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)
    for x, y, w, h in faces:
        print(x, y, w, h)

        # Recognizer Deep Learning Aligorithm


        # Rectangle Face Border Detection
        color = (255, 0 , 0)
        stroke = 2
        endCoordinateX = x + w
        endCoordinateY = y + h
        cv2.rectangle(frame, (x, y), (endCoordinateX, endCoordinateY), color, stroke)

    # Main Frame
    cv2.imshow('Main Frame', frame)
    # cv2.imshow('Grayscale Frame', grayscale)

    if cv2.waitKeyEx(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()