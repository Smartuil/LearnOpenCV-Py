import cv2 as cv

capture = cv.VideoCapture(0)
detector = cv.CascadeClassifier("D:/opencv/opencv/build/etc/lbpcascades/lbpcascade_frontalface_improved.xml")
while True:
    ret, image = capture.read()
    if ret is True:
        cv.imshow("frame", image)
        faces = detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=1,
                                          minSize=(30, 30), maxSize=(120, 120))
        for x, y, width, height in faces:
            cv.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2, cv.LINE_8, 0)
        cv.imshow("faces", image)
        c = cv.waitKey(50)
        if c == 27:
            break
    else:
        break

cv.destroyAllWindows()