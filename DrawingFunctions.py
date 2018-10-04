import numpy as np
import cv2

# To create a black image
img = np.zeros((512,512,3), np.uint8)

# To draw a diagonla blue line with 5 pixel thickness
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
#cv2.imshow('Drawing Practice', img)

img2 = cv2.rectangle(img,(384,0),(510,128),(0,255,0), 3)
img3 = cv2.circle(img,(447,63),63,(0,0,255),-1)
img4 = cv2.ellipse(img,(256,256),(100,50),0,0,180,(0,255,0),-1)

# Drawing multiple lines
pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
pts = pts.reshape((-1,1,2))
img5 = cv2.polylines(img, [pts], True,(0,255,255))

#Adding Text to Image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'Hi Champ!', (10,500),font, 3,(255,255,255),2,cv2.LINE_AA)

cv2.imshow('Rectangle Practice', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()