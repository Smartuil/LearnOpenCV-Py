import numpy as np
import cv2 as cv


def process(image, opt=1):
    # Detecting corners
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, 100, 0.05, 10)
    print(len(corners))
    for pt in corners:
        print(pt)
        b = np.random.random_integers(0, 256)
        g = np.random.random_integers(0, 256)
        r = np.random.random_integers(0, 256)
        x = np.int32(pt[0][0])
        y = np.int32(pt[0][1])
        cv.circle(image, (x, y), 5, (int(b), int(g), int(r)), 2)
    # output
    return image


src = cv.imread("D:/images/box.bmp")
cv.imshow("input", src)
result = process(src)
cv.imshow('result', result)
cv.waitKey(0)
cv.destroyAllWindows()
"""
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv.imwrite("D:/input.png", frame)
    cv.imshow('input', frame)
    result = process(frame)
    cv.imshow('result', result)
    k = cv.waitKey(5)&0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
"""