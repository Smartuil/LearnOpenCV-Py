import cv2 as cv

src = cv.imread("D:/images/test.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
cv.waitKey(0)
cv.destroyAllWindows()