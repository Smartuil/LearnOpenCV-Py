import cv2 as cv

box = cv.imread("D:/images/box.png");
box_in_sence = cv.imread("D:/images/box_in_scene.png");
cv.imshow("box", box)
cv.imshow("box_in_sence", box_in_sence)

# 创建AKAZE特征检测器
akaze = cv.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(box,None)
kp2, des2 = akaze.detectAndCompute(box_in_sence,None)

# 暴力匹配
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

# 绘制匹配
result = cv.drawMatches(box, kp1, box_in_sence, kp2, matches, None)
cv.imshow("orb-match", result)
cv.waitKey(0)
cv.destroyAllWindows()