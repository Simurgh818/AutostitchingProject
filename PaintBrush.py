# How to handle mouse events in OpenCV
# cv2.setMouseCallback() function

import cv2
events = [i for i in dir(cv2) if 'EVENT' in i]
print events