import cv2
import numpy as np 
from matplotlib import pyplot as plt

img1 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell1.tif',-1)
img2 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell2.tif',-1)

row, col = img1.shape


overLapCorrCoef = np.zeros((col),np.uint8)
overLapCorrCoef= np.corrcoef(img1[:,116:1:-1], img2[:,1:116], rowvar=False)[:,116]


plt.subplot(131),plt.imshow(img1,cmap = 'gray'),plt.title('Image 1')
plt.xticks([0,59,116]),plt.yticks([])

plt.subplot(132),plt.imshow(img2,cmap = 'gray'), plt.title('Image 2')
plt.xticks([0,59,116]),plt.yticks([])

plt.subplot(133),plt.plot(overLapCorrCoef)
plt.title('Correlation Coefficient'), plt.xticks([0,59,116,175,230]),plt.yticks([0,0.25,0.50,0.75,1])

plt.show()