import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('C:/Users/Sina/Dropbox (Gladstone)/Images/cell3.tif',-1)

row, col = img1.shape
print(img1)
# img1_array = np.zeros((row, col), np.uint8)
img1_array = np.array(img1, dtype=np.uint64)
print("int64 array is: ", img1_array)


plt.figure("Raw image")
plt.subplot(121)
plt.imshow(img1, cmap='gray')
plt.subplot(122)
plt.imshow(img1_array, cmap='gray')
plt.show()