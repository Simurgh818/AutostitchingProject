import numpy as np
import cv2

img = cv2.imread('/home/sinadabiri/Dropbox/s200_sina.dabiri.jpg',-1)
cv2.imshow('The Champ',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('/home/sinadabiri/Dropbox/s200_sina.dabiri.png',img)
    cv2.destroyAllWindows()