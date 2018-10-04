import cv2
import numpy as np 
from matplotlib import pyplot as plot

img1 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell1.tif',0)
img2 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell2.tif',0)

row, col = img1.shape

# Finding the optimal size for FFT (up to 4x faster)
nrow = cv2.getOptimalDFTSize(row)
ncol = cv2.getOptimalDFTSize(col)

print "height =" ,row, "width = " , col 
print "optimal row =", nrow, "optimal col = ", ncol

nimg1 = np.zeros((nrow,ncol))
nimg2 = np.zeros((nrow,ncol))
nimg1 [:row,:col]=img1
nimg2 [:row,:col]=img2


centerRow, centerCol = row/2, col/2

i=0
j = 0
overLapSum = np.zeros((nrow,ncol),np.uint8)
overLapAvg = np.zeros((nrow,ncol),np.uint8)

# Define a function for 2D image FFT
def ImageFFT(img1,row, col,centerRow, centerCol):
	# Calculating 2D FFT
	imgDFT = cv2.dft(np.float32(nimg1), flags=cv2.DFT_COMPLEX_OUTPUT)

	# shifting the DC component to the center of the image
	DFTShift = np.fft.fftshift(imgDFT)
	
	# calculating the magnitute of the complex numbers of DFTShift
	magnitudeSpectrum = 20*np.log(cv2.magnitude(DFTShift[:,:,0],DFTShift[:,:,1]))
	

	# Low pass filtering: creating a mask with high value of 1 at low freq and 
	# 0 at high freq
	freqMask = np.zeros((row,col,2),np.uint8)
	freqMask[centerRow-30:centerRow+30, centerCol-30:centerCol+30]=1

	#applying the mask
	DFTShiftMasked = DFTShift*freqMask

	#Inverse FFT
	freq_iFFT_shift = np.fft.ifftshift(DFTShiftMasked)
	img_iFFT = cv2.idft(freq_iFFT_shift)
	
	img_iFFT_mag = cv2.magnitude(img_iFFT[:,:,0],img_iFFT[:,:,1])

	# High Pass filtering: setting a 60x60 center rectangle to block low freq

	DFTShift[centerRow-15:centerRow+15, centerCol-15:centerCol+15]=0
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

def OverLayFFT(img1,img2):
	global overlay_iFFT_mag

	row1, col1 = img1.shape
	row2, col2 = img2.shape

	centerRow1, centerCol1 = row1/2, col1/2
	centerRow2, centerCol2 = row2/2, col2/2

	# Calculating 2D FFT
	img1DFT = cv2.dft(np.float32(nimg1), flags=cv2.DFT_COMPLEX_OUTPUT)
	img2DFT = cv2.dft(np.float32(nimg2), flags=cv2.DFT_COMPLEX_OUTPUT)

	# shifting the DC component to the center of the image
	DFTShift1 = np.fft.fftshift(img1DFT)
	DFTShift2 = np.fft.fftshift(img2DFT)
	
	# calculating the magnitute of the complex numbers of DFTShift
	img1_magnitudeSpectrum = 20*np.log(cv2.magnitude(DFTShift1[:,:,0],DFTShift1[:,:,1]))
	img2_magnitudeSpectrum = 20*np.log(cv2.magnitude(DFTShift2[:,:,0],DFTShift2[:,:,1]))

	# Low pass filtering: creating a mask with high value of 1 at low freq and 
	# 0 at high freq
	freqMask = np.zeros((row1,col1,2),np.uint8)
	freqMask[centerRow1-30:centerRow1+30, centerCol1-30:centerCol1+30]=1

	#applying the mask
	DFTShift1_Masked = DFTShift1*freqMask
	DFTShift2_Masked = DFTShift2*freqMask

	#Inverse FFT
	freq1_iFFT_shift = np.fft.ifftshift(DFTShift1_Masked)
	freq2_iFFT_shift = np.fft.ifftshift(DFTShift2_Masked)
	overlay_freq_iFFT_shift = freq1_iFFT_shift + freq2_iFFT_shift

	overlay_iFFT = cv2.idft(overlay_freq_iFFT_shift)
	overlay_iFFT_mag = cv2.magnitude(overlay_iFFT[:,:,0],overlay_iFFT[:,:,1])


	plot.subplot(141),plot.imshow(img1,cmap = 'gray'),plot.title('Image 1')
	plot.xticks([]),plot.yticks([])

	plot.subplot(142),plot.imshow(img2,cmap = 'gray'), plot.title('Image 2')
	plot.xticks([]),plot.yticks([])

	plot.subplot(143),plot.imshow(overlay_iFFT_mag,cmap = 'gray')
	plot.title('Overlay after LPF'), plot.xticks([]),plot.yticks([])

	plot.subplot(144),plot.imshow(overlay_iFFT_mag)
	plot.title('JET color LPF'), plot.xticks([]),plot.yticks([])

	plot.show()

	# return overlay_iFFT_mag

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



plot.subplot(131),plot.imshow(img1,'gray'),plot.title('Neuro')
plot.xticks([]),plot.yticks([])
plot.subplot(132),plot.imshow(overLapSum,'gray'),plot.title('Over Lap Add')
plot.xticks([]),plot.yticks([])
plot.subplot(133),plot.imshow(overLapAvg,'gray'),plot.title('Over Lap Avg')
plot.xticks([]),plot.yticks([])
plot.show()

OverLayFFT(img1,img2)

ImageFFT(img1,row,col,centerRow, centerCol)