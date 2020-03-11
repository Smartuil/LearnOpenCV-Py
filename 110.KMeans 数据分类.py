import numpy as np
import cv2
from matplotlib import pyplot as plt

X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))
pts = np.vstack((X,Y))

# 初始化数据
data = np.float32(pts)
print(data.shape)

# 定义停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(data,2,None,criteria,2,cv2.KMEANS_RANDOM_CENTERS)
print(len(label))
print(center)

# 获取不同标签的点
A = data[label.ravel()==0]
B = data[label.ravel()==1]

# Plot the data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()