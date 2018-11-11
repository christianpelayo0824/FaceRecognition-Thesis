import cv2

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Main Frame', frame)
    cv2.imshow('GrayScale', grayscale)

    if cv2.waitKeyEx(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
