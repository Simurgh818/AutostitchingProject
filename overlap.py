import cv2
import numpy as np 
from matplotlib import pyplot as plot

img1 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell1.tif',0)
img2 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell2.tif',0)

row, col = img1.shape
print "height =" ,row, "width = " , col 
centerRow, centerCol = row/2, col/2

i=0
j = 0
overLapSum = np.zeros((row,col),np.uint8)
overLapAvg = np.zeros((row,col),np.uint8)

# Define a function for 2D image FFT
def overLapFFT(img1,row, col,centerRow, centerCol):
	# Calculating 2D FFT
	imgDFT = cv2.dft(np.float32(img1), flags=cv2.DFT_COMPLEX_OUTPUT)
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

	# High Pass filtering: setting a 60x60 center rectangle to block low freq

	DFTShift[centerRow-30:centerRow+30, centerCol-30:centerCol+30]=0
	#inverse FFT
	freq_iFFT_shift_HPF = np.fft.ifftshift(DFTShift)
	img_iFFT_HPF = cv2.idft(freq_iFFT_shift_HPF)
	img_iFFT_HPF_mag = cv2.magnitude(img_iFFT_HPF[:,:,0], img_iFFT_HPF[:,:,1])

	plot.subplot(231),plot.imshow(img1,cmap = 'gray'),plot.title('Image')
	plot.xticks([]),plot.yticks([])

	plot.subplot(232),plot.imshow(img_iFFT_mag,cmap = 'gray')
	plot.title('Img after LPF'), plot.xticks([]),plot.yticks([])

	plot.subplot(233),plot.imshow(img_iFFT_HPF_mag,cmap = 'gray')
	plot.title('Img after HPF'), plot.xticks([]),plot.yticks([])

	plot.subplot(234),plot.imshow(img_iFFT_mag)
	plot.title('JET color LPF'), plot.xticks([]),plot.yticks([])

	plot.subplot(235),plot.imshow(img_iFFT_HPF_mag)
	plot.title('JET color HPF'), plot.xticks([]),plot.yticks([])

	plot.show()


for i in range(row):
	if i < row:
		for j in range (col):
			if j < col:
				overLapSum[i,j]= img1[i,j]+img2[i, j]
				overLapAvg[i,j]= (img1[i,j]+img2[i, j])/2

			else:
				break
		
	else:
		break
print "column = ", j
print "row = ", i

print (overLapSum)
print (overLapAvg)



plot.subplot(141),plot.imshow(img1,'gray'),plot.title('Neuro')
plot.xticks([]),plot.yticks([])
plot.subplot(142),plot.imshow(overLapSum,'gray'),plot.title('Over Lap Add')
plot.xticks([]),plot.yticks([])
plot.subplot(143),plot.imshow(overLapAvg,'gray'),plot.title('Over Lap Avg')
plot.xticks([]),plot.yticks([])

plot.show()

overLapFFT(img1,row,col,centerRow, centerCol)