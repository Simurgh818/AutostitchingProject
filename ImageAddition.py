import numpy as np 
import cv2


x = np.uint16([250])
y = np.uint16([10])
print cv2.add(x,y) #although 250+10=260, max color is 255. 
# OpenCV addition is a saturated opearion, while Numpy addition is a modulo opeation.
# OpenCV is better.

print x+y # 250+10=260 , 260 % 256 = 4

img1 = cv2.imread('/home/sinadabiri/Dropbox/Images/BrainReading.bmp',-1)
img2 = cv2.imread('/home/sinadabiri/Dropbox/Images/s200_sina.dabiri.png',-1)
img1_resized = cv2.resize(img1, (img2.shape[1],img2.shape[0]))
# height, width, channels = img1.shape
# print height, width, channels

while(1):
	cv2.namedWindow('Brain', cv2.WINDOW_NORMAL)
	cv2.imshow('Brain',img1)
	cv2.namedWindow('Sina', cv2.WINDOW_NORMAL)
	cv2.imshow('Sina',img2)
	k = cv2.waitKey(1) & 0xFF # 0xFF is hex for integer 255
	if k == 32:
		break


dst = cv2.addWeighted(img1_resized,0.5, img2,0.5,0)
 #the two image needs to be the same size?
cv2.imshow('dst',dst)

cv2.waitKey(0)
cv2.destroyAllWindows()