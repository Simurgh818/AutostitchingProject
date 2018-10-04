import numpy as np 
import cv2
import matplotlib.pyplot as plot

img = cv2.imread('/home/sinadabiri/Dropbox/Images/s200_sina.dabiri.png',0)
row, col = img.shape
# Finding the center, to be used to make a rectangular for high pass filtering
# which will remove noise 

centerRow,centerCol = row/2,col/2

# Calculating 2D FFT
imgDFT = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
print (imgDFT)
# shifting the DC component to the center of the image
DFTShift = np.fft.fftshift(imgDFT)
print (DFTShift)
# calculating the magnitute of the complex numbers of freShift
magnitudeSpectrum = 20*np.log(cv2.magnitude(DFTShift[:,:,0],DFTShift[:,:,1]))
print (magnitudeSpectrum)

# Low pass filtering: creating a mask with high value of 1 at low freq and 
# 0 at high freq
freqMask = np.zeros((row,col,2),np.uint8)
freqMask[centerRow-30:centerRow+30, centerCol-30:centerCol+30]=1

#applying the mask
DFTShiftMasked = DFTShift*freqMask

#Inverse FFT
freq_iFFT_shift = np.fft.ifftshift(DFTShiftMasked)
img_iFFT = cv2.idft(freq_iFFT_shift)
print(img_iFFT)
img_iFFT_mag = cv2.magnitude(img_iFFT[:,:,0],img_iFFT[:,:,1])


plot.subplot(141),plot.imshow(img,cmap = 'gray'),plot.title('Image')
plot.xticks([]),plot.yticks([])

plot.subplot(142),plot.imshow(magnitudeSpectrum,cmap = 'gray')
plot.title('Mag Spect'), plot.xticks([]),plot.yticks([])

plot.subplot(143),plot.imshow(img_iFFT_mag,cmap = 'gray')
plot.title('Img after LPF'), plot.xticks([]),plot.yticks([])

plot.subplot(144),plot.imshow(img_iFFT_mag)
plot.title('Img boundary'), plot.xticks([]),plot.yticks([])
plot.show()