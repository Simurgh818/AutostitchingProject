import cv2
import numpy as np 
from matplotlib import pyplot as plt

img1 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell3.tif',-1)
img2 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell3.tif',-1)


row, col = img1.shape
print "Image 1 height =" ,row, "Image 1 width = " , col 
row2, col2 = img2.shape
print "Image 2 height =" ,row2, "Image 2 width = " , col2 

i = 0
j = 0
overLapCorrCoef = np.zeros((col),np.uint16)
overLapCorrCoefAvg = np.zeros((col),np.uint16)
overLapCorrCoefFFT = np.zeros((col),np.float32)
overLapCorrCoefLPF = np.zeros((col),np.float32)
overLapCorrCoefHPF = np.zeros((col),np.float32)

img1_Avg = np.zeros((row-1,col-1),np.uint16)
img2_Avg = np.zeros((row2-1,col2-1),np.uint16)

overLapCorrCoef= np.corrcoef(img1[:,(col-1):1:-1], img2[:,1:(col2-1)], rowvar=False)[:,(col-1)]

pixelNumbersToAverage = 50
pixelincrement = pixelNumbersToAverage/2-1

def PixelAveraging(img1,img2, pixelNumbersToAverage,pixelincrement):
	# compute the average of adjacent columns
	for i in range (row):
		if i<(row-1): 
			
			for j in range(col):
				if j<(col-1):
					img1_Avg[i:i+pixelincrement,j:j+pixelincrement]=np.average([img1[i:i+pixelincrement,j:j+pixelincrement]])
					img2_Avg[i:i+pixelincrement,j:j+pixelincrement]=np.average([img2[i:i+pixelincrement,j:j+pixelincrement]])
					j=j+pixelNumbersToAverage/2
				else:
					break
			i=i+pixelNumbersToAverage/2
		else:
			break
	return img1_Avg, img2_Avg

PixelAveraging(img1,img2,pixelNumbersToAverage,pixelincrement)

overLapCorrCoefAvg= np.corrcoef(img1_Avg[1:(row-1),(col-1):1:-1], img2_Avg[1:(row2-1),1:(col2-1)], 
	rowvar=False)[:,(col-1)]


def OverLayFFT(img1,img2):
	global ang1, ang2, img1DFT, img2DFT, DFTShift1, DFTShift2, img1_magnitudeSpectrum, img2_magnitudeSpectrum, overlayFFT_mag, img1_LPF_iFFT_mag,img2_LPF_iFFT_mag, img1_HPF_iFFT_mag, img2_HPF_iFFT_mag, overlayFFT_phase, overlayFFT_phase2, polarDiff,x,y

	row1, col1 = img1.shape
	row2, col2 = img2.shape

	centerRow1, centerCol1 = row1/2, col1/2
	centerRow2, centerCol2 = row2/2, col2/2

	centerRectangle = 15

	# Calculating 2D FFT
	img1FFT2 = np.fft.fft2(img1)
	img2FFT2 = np.fft.fft2(img2)

	# WORK IN PROGRESS:

	polarDiff = img1FFT2 - img2FFT2


	# shifting the DC component to the center of the image
	FFT2Shift1 = np.fft.fftshift(img1FFT2)
	FFT2Shift2 = np.fft.fftshift(img2FFT2)
		# Need to verify DFTShift1[:,:,1] is angle or DFTShift1[:,:,0]? DFTShift1[:,:,1] is the angle


	ang1 = np.angle(img1FFT2)
	ang2 = np.angle(img2FFT2)
	
	
	# ang1 = cv2.phase(img1DFT[:,:,0], img1DFT[:,:,1])
	# ang2 = cv2.phase(img2DFT[:,:,0], img2DFT[:,:,1])

	
	# overlayFFT_phase2

	# m,n = overlayFFT_phase2.shape
	# print(m,n)

	
	# calculating the magnitute of the complex numbers of DFTShift
	img1_magnitudeSpectrum = 20*np.log(np.abs(FFT2Shift1))
	img2_magnitudeSpectrum = 20*np.log(np.abs(FFT2Shift2))

	#TODO: try adding the manitudes of the two image in freq domain for overlay FFT
	overlayFFT_mag = img2_magnitudeSpectrum - img1_magnitudeSpectrum

	overlayFFT_phase = ang2-ang1
	print(overlayFFT_phase)
	x,y = cv2.polarToCart(overlayFFT_mag,overlayFFT_phase)
	print(x)
	# cv2.determinant(ang1) - cv2.determinant(ang2)
	

	# print (overlayFFT_phase)

	# END OF WORK IN PROGRESS
	# ------------------------------------------------------------------------------
	
	# Low pass filtering: creating a mask with high value of 1 at low freq and 
	# 0 at high freq
	freqMask1_LPF = np.zeros((row1,col1,2),np.float32)
	freqMask2_LPF = np.zeros((row2,col2,2),np.float32)
	freqMask1_LPF[centerRow1-centerRectangle:centerRow1+centerRectangle, centerCol1-centerRectangle:centerCol1+centerRectangle]=1
	freqMask2_LPF[centerRow2-centerRectangle:centerRow2+centerRectangle, centerCol2-centerRectangle:centerCol2+centerRectangle]=1

	img1DFT = cv2.dft(np.float32(img1), flags=cv2.DFT_COMPLEX_OUTPUT)
	img2DFT = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)

	DFTShift1 = np.fft.fftshift(img1DFT)
	DFTShift2 = np.fft.fftshift(img2DFT)
	#applying the mask for LPF
	DFTShift1_LPF_Masked = DFTShift1*freqMask1_LPF
	DFTShift2_LPF_Masked = DFTShift2*freqMask2_LPF

	#Inverse FFT
	img1_LPF_iFFT_shift  = np.fft.ifftshift(DFTShift1_LPF_Masked)
	img2_LPF_iFFT_shift  = np.fft.ifftshift(DFTShift2_LPF_Masked)

	img1_LPF_iFFT = cv2.idft(img1_LPF_iFFT_shift)
	img2_LPF_iFFT = cv2.idft(img2_LPF_iFFT_shift)
	img1_LPF_iFFT_mag = cv2.magnitude(img1_LPF_iFFT[:,:,0],img1_LPF_iFFT[:,:,1])
	img2_LPF_iFFT_mag = cv2.magnitude(img2_LPF_iFFT[:,:,0],img2_LPF_iFFT[:,:,1])

	# High Pass filtering: setting a 15x15 center rectangle to block low freq

	DFTShift1[centerRow1-centerRectangle:centerRow1+centerRectangle, centerCol1-centerRectangle:centerCol1+centerRectangle]=0
	DFTShift2[centerRow2-centerRectangle:centerRow2+centerRectangle, centerCol2-centerRectangle:centerCol2+centerRectangle]=0
	
	#inverse FFT
	img1_HPF_iFFT_shift = np.fft.ifftshift(DFTShift1)
	img2_HPF_iFFT_shift = np.fft.ifftshift(DFTShift2)

	img1_HPF_iFFT = cv2.idft(img1_HPF_iFFT_shift)
	img2_HPF_iFFT = cv2.idft(img2_HPF_iFFT_shift)
	
	img1_HPF_iFFT_mag = cv2.magnitude(img1_HPF_iFFT[:,:,0], img1_HPF_iFFT[:,:,1])
	img2_HPF_iFFT_mag = cv2.magnitude(img2_HPF_iFFT[:,:,0], img2_HPF_iFFT[:,:,1])

	return ang1, ang2, img1DFT, img2DFT, DFTShift1, DFTShift2, img1_magnitudeSpectrum, img2_magnitudeSpectrum, overlayFFT_mag, img1_LPF_iFFT_mag,img2_LPF_iFFT_mag, img1_HPF_iFFT_mag, img2_HPF_iFFT_mag, overlayFFT_phase, polarDiff,x,y;



OverLayFFT(img1,img2)
# print(DFTShift1)

overLapCorrCoefFFT= np.corrcoef(ang1[1:(row-1),(col-1):1:-1],
 ang2[1:(row2-1),1:(col2-1)], rowvar=False)[:,(col-1)]
overLapCorrCoefLPF= np.corrcoef(img1_LPF_iFFT_mag[1:(row-1),(col-1):1:-1], 
	img2_LPF_iFFT_mag[1:(row2-1),1:(col2-1)], rowvar=False)[:,(col-1)]
overLapCorrCoefHPF= np.corrcoef(img1_HPF_iFFT_mag[1:(row-1),(col-1):1:-1], img2_HPF_iFFT_mag[1:(row2-1),1:(col2-1)], rowvar=False)[:,(col-1)]

# print(overLapCorrCoefAvg)

plt.figure("Spatial domain")
plt.subplot(321),plt.imshow(img1,cmap = 'gray'),plt.title('Image 1')
plt.xticks([0,62,124]),plt.yticks([])
plt.subplot(323),plt.imshow(img2,cmap = 'gray'), plt.title('Image 2')
plt.xticks([0,62,124]),plt.yticks([])
plt.subplot(325),plt.plot(overLapCorrCoef)
plt.title('Corr Coef'), plt.xticks(np.arange(0,249,62)),plt.yticks(np.arange(-0.8,1.3,0.2))
plt.subplot(322),plt.imshow(img1_Avg,cmap = 'gray'),plt.title('Image 1 50 pixel Avg')
plt.xticks([0,62,124]),plt.yticks([])
plt.subplot(324),plt.imshow(img2_Avg,cmap = 'gray'), plt.title('Image 2 50 pixel Avg')
plt.xticks([0,62,124]),plt.yticks([])
plt.subplot(326),plt.plot(overLapCorrCoefAvg)
plt.title('Avg Corr Coef'), plt.xticks(np.arange(0,249,62)),plt.yticks(np.arange(-0.8,1.3,0.2))
plt.show()

plt.figure("Freq domain")
plt.subplot(331),plt.imshow(ang1)
plt.title('img1 FFT phase spectrum')
plt.subplot(334),plt.imshow(overlayFFT_phase)
plt.title('img2-img1 FFT phase spectrum')
plt.subplot(337),plt.plot(overLapCorrCoefFFT), plt.title('')
plt.xticks(np.arange(0,249,62)), plt.yticks(np.arange(-0.8,1.3,0.2))

plt.subplot(332),plt.imshow(img1_LPF_iFFT_mag,cmap = 'gray')
plt.title('img1 iFFT LPF'), plt.xticks([0,62,124]),plt.yticks([])
plt.subplot(335),plt.imshow(img2_LPF_iFFT_mag,cmap = 'gray')
plt.title('img2 iFFT LPF'), plt.xticks([0,62,124]),plt.yticks([])
plt.subplot(338),plt.plot(overLapCorrCoefLPF), plt.title('LPF Corr Coef'), 
plt.xticks(np.arange(0,249,62)), plt.yticks(np.arange(-0.8,1.3,0.2))

plt.subplot(333),plt.imshow(img1_HPF_iFFT_mag)
plt.title('img1 iFFT HPF'), plt.xticks([0,62,124]),plt.yticks([])
plt.subplot(336),plt.imshow(img2_HPF_iFFT_mag)
plt.title('img2 iFFT HPF'), plt.xticks([0,62,124]),plt.yticks([])
plt.subplot(339),plt.plot(overLapCorrCoefHPF), plt.title('HPF Corr Coef'), 
plt.xticks(np.arange(0,249,62)), plt.yticks(np.arange(-0.8,1.3,0.2))
plt.show()