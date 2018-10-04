import cv2
import numpy as np 
from matplotlib import pyplot as plt

img1 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell1.tif',0)
img2 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell2.tif',0)

row, col = img1.shape

# row1, col1 = img1.shape
# row2, col2 = img2.shape
print "height =" ,row, "width = " , col 
# centerRow1, centerCol1 = row1/2, col1/2
# centerRow2, centerCol2 = row2/2, col2/2
width = col-1

i=0
j = 0
overLapCorrCoef = np.zeros((col),np.uint8)
overLapCorrCoef= np.corrcoef(img1[:,116:1:-1], img2[:,1:116], rowvar=False)[116,:]
# overLapCorrCoef= np.corrcoef(img1[:,116:1:-1], img2[:,1:116], rowvar=True)
# overLapCorrCoef= np.corrcoef(img1[:,116], img2[:,1], rowvar=True)
print (overLapCorrCoef)
print (np.size(overLapCorrCoef))
# print "image 1 ", img1[:,116], "image 2 ", img2[:,1]

# for i in range(col):
# 	if i < col:

# 		overLapCorrCoef[:,i]= np.corrcoef(img1[:,width],img2[:,i], rowvar=False)[1,0]

# 		print (overLapCorrCoef[i])
# 		# width = width -1
# 	else:
# 		break

plt.subplot(131),plt.imshow(img1,cmap = 'gray'),plt.title('Image 1')
plt.xticks([]),plt.yticks([])

plt.subplot(132),plt.imshow(img2,cmap = 'gray'), plt.title('Image 2')
plt.xticks([]),plt.yticks([])

plt.subplot(133),plt.plot(overLapCorrCoef)
plt.title('Correlation Coefficient'), plt.xticks([col]),plt.yticks([0,0.25,0.50,0.75,1])


plt.show()



