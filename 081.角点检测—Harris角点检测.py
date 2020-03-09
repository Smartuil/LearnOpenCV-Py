import numpy as np
import cv2 as cv


def process(image, opt=1):
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > 80:
                b = np.random.random_integers(0, 256)
                g = np.random.random_integers(0, 256)
                r = np.random.random_integers(0, 256)
                cv.circle(image, (j, i), 5, (int(b), int(g), int(r)), 2)
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