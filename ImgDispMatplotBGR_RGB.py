import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/sinadabiri/Dropbox/s200_sina.dabiri.jpg',-1)
# img2 = img[:,:,::-1]
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(121); plt.imshow(img)
plt.subplot(122);plt.imshow(img2)
plt.show()

cv2.imshow('BGR image', img)
cv2.imshow('RGB image', img2)


cv2.waitKey(0)
cv2.destroyAllWindows()

