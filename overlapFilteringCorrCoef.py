import cv2
import numpy as np 
from matplotlib import pyplot as plt

img1 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell1.tif',-1)
img2 = cv2.imread('/home/sinadabiri/Dropbox/Images/cell2.tif',-1)


row, col = img1.shape
print "height =" ,row, "width = " , col 
row2, col2 = img2.shape

i = 0
j = 0
overLapCorrCoef = np.zeros((col),np.uint16)
overLapCorrCoefAvg = np.zeros((col),np.uint16)
overLapCorrCoefFFT = np.zeros((col),np.float32)
overLapCorrCoefLPF = np.zeros((col),np.float32)
overLapCorrCoefHPF = np.zeros((col),np.float32)

img1_Avg = np.zeros((row-1,col-1),np.uint16)
img2_Avg = np.zeros((row2-1,col2-1),np.uint16)

overLapCorrCoef= np.corrcoef(img1[:,116:1:-1], img2[:,1:116], rowvar=False)[:,116]

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

overLapCorrCoefAvg= np.corrcoef(img1_Avg[1:113,116:1:-1], img1_Avg[1:113,1:116], rowvar=False)[:,116]


def OverLayFFT(img1,img2):
	global img1DFT, img2DFT, DFTShift1, DFTShift2, img1_magnitudeSpectrum, img2_magnitudeSpectrum, overlayFFT_mag, img1_LPF_iFFT_mag,img2_LPF_iFFT_mag, img1_HPF_iFFT_mag, img2_HPF_iFFT_mag

	row1, col1 = img1.shape
	row2, col2 = img2.shape

	centerRow1, centerCol1 = row1/2, col1/2
	centerRow2, centerCol2 = row2/2, col2/2

	# Calculating 2D FFT
	img1DFT = cv2.dft(np.float32(img1), flags=cv2.DFT_COMPLEX_OUTPUT)
	img2DFT = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)
	# magImg1DFT, angImg1DFT = (img1DFT) 
	print (img1DFT)

	# shifting the DC component to the center of the image
	DFTShift1 = np.fft.fftshift(img1DFT)
	DFTShift2 = np.fft.fftshift(img2DFT)
	
	# calculating the magnitute of the complex numbers of DFTShift
	img1_magnitudeSpectrum = 20*np.log(cv2.magnitude(DFTShift1[:,:,0],DFTShift1[:,:,1]))
	img2_magnitudeSpectrum = 20*np.log(cv2.magnitude(DFTShift2[:,:,0],DFTShift2[:,:,1]))

	#TODO: try adding the manitudes of the two image in freq domain for overlay FFT
	overlayFFT_mag = img1_magnitudeSpectrum + img2_magnitudeSpectrum

	# Low pass filtering: creating a mask with high value of 1 at low freq and 
	# 0 at high freq
	freqMask1_LPF = np.zeros((row1,col1,2),np.float32)
	freqMask2_LPF = np.zeros((row2,col2,2),np.float32)
	freqMask1_LPF[centerRow1-30:centerRow1+30, centerCol1-30:centerCol1+30]=1
	freqMask2_LPF[centerRow2-30:centerRow2+30, centerCol2-30:centerCol2+30]=1

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

	# High Pass filtering: setting a 30x30 center rectangle to block low freq

	DFTShift1[centerRow1-15:centerRow1+15, centerCol1-15:centerCol1+15]=0
	DFTShift2[centerRow2-15:centerRow2+15, centerCol2-15:centerCol2+15]=0
	
	#inverse FFT
	img1_HPF_iFFT_shift = np.fft.ifftshift(DFTShift1)
	img2_HPF_iFFT_shift = np.fft.ifftshift(DFTShift2)

	img1_HPF_iFFT = cv2.idft(img1_HPF_iFFT_shift)
	img2_HPF_iFFT = cv2.idft(img2_HPF_iFFT_shift)
	
	img1_HPF_iFFT_mag = cv2.magnitude(img1_HPF_iFFT[:,:,0], img1_HPF_iFFT[:,:,1])
	img2_HPF_iFFT_mag = cv2.magnitude(img2_HPF_iFFT[:,:,0], img2_HPF_iFFT[:,:,1])

	return img1DFT, img2DFT, DFTShift1, DFTShift2, img1_magnitudeSpectrum, img2_magnitudeSpectrum, overlayFFT_mag, img1_LPF_iFFT_mag,img2_LPF_iFFT_mag, img1_HPF_iFFT_mag, img2_HPF_iFFT_mag;



OverLayFFT(img1,img2)
# print(DFTShift1)

overLapCorrCoefFFT= np.corrcoef(img1_magnitudeSpectrum[1:113,116:1:-1], img1_magnitudeSpectrum[1:113,1:116], rowvar=False)[:,116]
overLapCorrCoefLPF= np.corrcoef(img1_LPF_iFFT_mag[1:113,116:1:-1], img2_LPF_iFFT_mag[1:113,1:116], rowvar=False)[:,116]
overLapCorrCoefHPF= np.corrcoef(img1_HPF_iFFT_mag[1:113,116:1:-1], img2_HPF_iFFT_mag[1:113,1:116], rowvar=False)[:,116]

# print(overLapCorrCoefAvg)

plt.figure("Time domain")
plt.subplot(321),plt.imshow(img1,cmap = 'gray'),plt.title('Image 1')
plt.xticks([0,59,116]),plt.yticks([])
plt.subplot(323),plt.imshow(img2,cmap = 'gray'), plt.title('Image 2')
plt.xticks([0,59,116]),plt.yticks([])
plt.subplot(325),plt.plot(overLapCorrCoef)
plt.title('Corr Coef'), plt.xticks([0,59,116,175,230])
plt.subplot(322),plt.imshow(img1_Avg,cmap = 'gray'),plt.title('Image 1 Avg')
plt.xticks([0,59,116]),plt.yticks([])
plt.subplot(324),plt.imshow(img2_Avg,cmap = 'gray'), plt.title('Image 2 Avg')
plt.xticks([0,59,116]),plt.yticks([])
plt.subplot(326),plt.plot(overLapCorrCoefAvg)
plt.title('Avg Corr Coef'), plt.xticks([0,59,116,175,230])
plt.show()

plt.figure("Freq domain")
plt.subplot(331),plt.imshow(img1_magnitudeSpectrum,cmap = 'gray')
plt.title('img1 FFT mag spectrum'), plt.xticks([]),plt.yticks([])
plt.subplot(334),plt.imshow(img2_magnitudeSpectrum,cmap = 'gray')
plt.title('img2 FFT mag spectrum'), plt.xticks([]),plt.yticks([])
plt.subplot(337),plt.plot(overLapCorrCoefFFT)
plt.title('FFT mag Corr Coef'), plt.xticks([0,59,116,175,230])

plt.subplot(332),plt.imshow(img1_LPF_iFFT_mag)
plt.title('img1 iFFT LPF JET color'), plt.xticks([0,59,116]),plt.yticks([])
plt.subplot(335),plt.imshow(img2_LPF_iFFT_mag)
plt.title('img2 iFFT LPF JET color'), plt.xticks([0,59,116]),plt.yticks([])
plt.subplot(338),plt.plot(overLapCorrCoefLPF)
plt.title('LPF Corr Coef'), plt.xticks([0,59,116,175,230])

plt.subplot(333),plt.imshow(img1_HPF_iFFT_mag)
plt.title('img1 iFFT HPF JET color'), plt.xticks([0,59,116]),plt.yticks([])
plt.subplot(336),plt.imshow(img2_HPF_iFFT_mag)
plt.title('img2 iFFT HPF JET color'), plt.xticks([0,59,116]),plt.yticks([])
plt.subplot(339),plt.plot(overLapCorrCoefHPF)
plt.title('HPF Corr Coef'), plt.xticks([0,59,116,175,230])
plt.show()