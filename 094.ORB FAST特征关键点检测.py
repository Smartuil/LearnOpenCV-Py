import cv2 as cv
import numpy as np

src = cv.imread("D:/images/test1.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
orb = cv.ORB().create()
kps = orb.detect(src)
result = cv.drawKeypoints(src, kps, None, (0, 255, 0), cv.DrawMatchesFlags_DEFAULT)
cv.imshow("result", result)
cv.waitKey(0)
cv.destroyAllWindows()