import cv2 as cv
import numpy as np

src = cv.imread("D:/images/flower.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
sift = cv.xfeatures2d.SIFT_create()
kps = sift.detect(src)
result = cv.drawKeypoints(src, kps, None, (0, 0, 255), cv.DrawMatchesFlags_DEFAULT)
cv.imshow("sift-detector", result)
cv.waitKey(0)
cv.destroyAllWindows()