import numpy as np 
import cv2
import matplotlib.pyplot as plot

img = cv2.imread('/home/sinadabiri/Dropbox/Images/cell1.tif',0)
row, col = img.shape
# Finding the center, to be used to make a rectangular for high pass filtering
# which will remove noise  /home/sinadabiri/Dropbox/Images/s200_sina.dabiri.png

centerRow,centerCol = row/2,col/2

# Calculating 2D FFT
imgFreq = np.fft.fft2(img)
print (imgFreq)
# shifting the DC component to the center of the image
freqShift = np.fft.fftshift(imgFreq)
print (freqShift)
# calculating the magnitute of the complex numbers of freShift
magnitudeSpectrum = 20*np.log(np.abs(freqShift))
print (magnitudeSpectrum)

# High pass filtering: setting a 60x60 center rectangle as zero
freqShift[centerRow-15:centerRow+15, centerCol-15:centerCol+15]=0
#Inverse FFT
freq_iFFT_shift = np.fft.ifftshift(freqShift)
img_iFFT = np.fft.ifft2(freq_iFFT_shift)
print(img_iFFT)
img_iFFT_mag = np.abs(img_iFFT)




plot.subplot(141),plot.imshow(img,cmap = 'gray'),plot.title('Image')
plot.xticks([]),plot.yticks([])

plot.subplot(142),plot.imshow(magnitudeSpectrum,cmap = 'gray')
plot.title('Mag Spect'), plot.xticks([]),plot.yticks([])

plot.subplot(143),plot.imshow(img_iFFT_mag,cmap = 'gray')
plot.title('Img after HPF'), plot.xticks([]),plot.yticks([])

plot.subplot(144),plot.imshow(img_iFFT_mag)
plot.title('Img boundary'), plot.xticks([]),plot.yticks([])
plot.show()