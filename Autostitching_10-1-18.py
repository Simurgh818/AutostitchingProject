import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import array
from datetime import datetime

img = [0]*30
path = ' '

k = 1
# /home/sinadabiri/Dropbox/Images/cell/
for file in os.listdir('/home/sinadabiri/Dropbox/Images/robo6_2x2_103pixelOverlap'):
    if file.endswith(".tif"):

			# /home/sinadabiri/Dropbox/Images/cell/
    	path = str(os.path.join('/home/sinadabiri/Dropbox/Images/robo6_2x2_103pixelOverlap', file))

    	img[k] = cv2.imread(path, -1)
    	# print (img[i])
    	k+=1
        continue
    else:
        continue

    # if any('overlay' in x for x in channels):
    #     assert os.path.exists(overlays_path), 'Confirm path for overlays exists (%s)' % overlays_path

# robo6_2x2_103pixelOverlap/PID20180604_Robo6Test-TDP43-20180531_T5_60-0_B13_4_Epi-RFP_0.0_0_1.0.tif
# robo6_2x2_103pixelOverlap/PID20180604_Robo6Test-TDP43-20180531_T5_60-0_B13_3_Epi-RFP_0.0_0_1.0.tif
# img[1] = cv2.imread('/home/sinadabiri/Dropbox/Images/cell3.tif',-1)
# img[2] = cv2.imread('/home/sinadabiri/Dropbox/Images/cell3Shifted.tif',-1)


for x in xrange(1,10):
	pass


k=1

row, col = img[k+2].shape
print("Image 1 height =" ,row)
print("Image 1 width = " , col)

row2, col2 = img[k+1].shape
print("Image 2 height =" ,row2)
print("Image 2 width = " , col2)



overLapCorrCoef = np.zeros((col),np.uint16)


img1_Avg = np.zeros((row-1,col-1),np.uint16)
img2_Avg = np.zeros((row2-1,col2-1),np.uint16)

overLapCorrCoef= np.corrcoef(img[k+2][:,(col-1):1:-1], img[k+1][:,1:(col2-1)], rowvar=False)[:,(col-1)]

overLapCorrCoefVertical= np.corrcoef(img[k+2][(row-1):1:-1,:], img[k+1][1:(row2-1),:], rowvar=True)[(row-1),:]

deltaX= np.argmax(overLapCorrCoef[1:col/2])
print(col/2, overLapCorrCoef[1:col/2])
print("delta x is: ", deltaX, "local max of Horizontal Corr Coef is: ", np.amax(overLapCorrCoef[1:col/2]))

deltaY = np.argmax(overLapCorrCoefVertical[1:row/2])
print(row/2, overLapCorrCoefVertical[1:row/2])
print("delta y is: ", deltaY,"local max of Vertical Corr Coef is: ", np.amax(overLapCorrCoefVertical[1:row/2]))

# # ---------------------------------------------------
# # Geometric translation
# M = np.float32([[1,0,-(deltaX-1)],[0,1,(deltaY-1)]])
# img2_translated = cv2.warpAffine(img[k+1],M,(col2,row2))

# date_time = datetime.now().strftime("%m-%d-%Y_%H:%M")

# translated_image_path = '/home/sinadabiri/Dropbox/Images/cell3Translated_' + date_time +'.tif'

# cv2.imwrite(translated_image_path,img2_translated)
# # -------------------------------------------------

i = 0
j = 0
overLapCorrCoefAvg = np.zeros((col),np.uint16)
overLapCorrCoefFFT = np.zeros((col),np.float32)
overLapCorrCoefLPF = np.zeros((col),np.float32)
overLapCorrCoefHPF = np.zeros((col),np.float32)

pixelNumbersToAverage = 50
pixelincrement = pixelNumbersToAverage/2-1

def PixelAveraging(img1,img2, pixelNumbersToAverage,pixelincrement):
	# compute the average of adjacent columns
	for i in range (row):
		if i<(row-1):

			for j in range(col):
				if j<(col-1):
					img1_Avg[i:i+pixelincrement,j:j+pixelincrement]=np.average([img[k+2][i:i+pixelincrement,j:j+pixelincrement]])
					img2_Avg[i:i+pixelincrement,j:j+pixelincrement]=np.average([img[k+1][i:i+pixelincrement,j:j+pixelincrement]])
					j=j+pixelNumbersToAverage/2
				else:
					break
			i=i+pixelNumbersToAverage/2
		else:
			break
	return img1_Avg, img2_Avg;

PixelAveraging(img[k+2],img[k+1],pixelNumbersToAverage,pixelincrement)

overLapCorrCoefAvg= np.corrcoef(img1_Avg[1:(row-1),(col-1):1:-1],
	img2_Avg[1:(row2-1),1:(col2-1)], 	rowvar=False)[:,(col-1)]

# 0 for col

# for ind in range(col/2):
# 	if np.maximum(overLapCorrCoefAvg[:,ind]) > 0.5:
# 		print ("x : ", ind)
# 	else:
# 		break


def OverLayFFT(img1,img2):
	global ang1, ang2, img1DFT, img2DFT, DFTShift1, DFTShift2, img1_magnitudeSpectrum, img2_magnitudeSpectrum, overlayFFT_mag, img1_LPF_iFFT_mag,img2_LPF_iFFT_mag, img1_HPF_iFFT_mag, img2_HPF_iFFT_mag, overlayFFT_phase, overlayFFT_phase2, CartDiff, overlayFFT, overlayiFFT_mag,overlayiFFT,img2Corrected, img2PhaseCorrected

	row1, col1 = img[k+2].shape
	row2, col2 = img[k+1].shape

	centerRow1, centerCol1 = row1/2, col1/2
	centerRow2, centerCol2 = row2/2, col2/2

	centerRectangle = 15

	# Calculating 2D FFT
	img1FFT2 = np.fft.fft2(img[k+2])
	img2FFT2 = np.fft.fft2(img[k+1])

	# WORK IN PROGRESS:

	CartDiff = img2FFT2 - img1FFT2
	# print ("Cart Diff: ", CartDiff)
	img2FFT2 = img1FFT2 - CartDiff

	# shifting the DC component to the center of the image
	FFT2Shift1 = np.fft.fftshift(img1FFT2)
	FFT2Shift2 = np.fft.fftshift(img2FFT2)
		# Need to verify DFTShift1[:,:,1] is angle or DFTShift1[:,:,0]? DFTShift1[:,:,1] is the angle


	ang1 = np.angle(img1FFT2)
	ang2 = np.angle(img2FFT2)

	overlayFFT_phase = np.angle(img1FFT2)-np.angle(img2FFT2)
	# print("overlay FFT phase shift", overlayFFT_phase)

	img2PhaseCorrected = np.angle(FFT2Shift2) - overlayFFT_phase

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

	# img2MagCorrected = img1_magnitudeSpectrum + overlayFFT_mag

	# print(overlayFFT_phase)

	# Convert back to space domain and then graph the overlay diff



	# cv2.determinant(ang1) - cv2.determinant(ang2)

	overlayFFT = [img2_magnitudeSpectrum+img2PhaseCorrected*j]
	# print("overlay FFT: ", overlayFFT)



	overlayFFT_ishift = np.fft.ifftshift(overlayFFT)
	# overlayiFFT = np.fft.ifft2(overlayFFT_ishift)
	overlayiFFT = np.fft.ifft2(img2FFT2)

	img2Corrected = np.abs(overlayiFFT)
	# print("image 2 corrected: ", img2Corrected.shape, img2Corrected[:,:])


	# x,y = cv2.polarToCart(overlayFFT_ishift[0,:,:],overlayFFT_ishift[1,:,:])

	# overlayiFFT_mag = np.abs(overlayiFFT[0,:,:])
	# row4, col4 =overlayiFFT_mag.shape
	# print(row4, col4)

	# print("overlay iFFT Mag: ", overlayiFFT_mag)
	# print (overlayFFT_phase)

	# END OF WORK IN PROGRESS
	# ------------------------------------------------------------------------------

	# Low pass filtering: creating a mask with high value of 1 at low freq and
	# 0 at high freq
	freqMask1_LPF = np.zeros((row1,col1,2),np.float32)
	freqMask2_LPF = np.zeros((row2,col2,2),np.float32)
	freqMask1_LPF[centerRow1-centerRectangle:centerRow1+centerRectangle, centerCol1-centerRectangle:centerCol1+centerRectangle]=1
	freqMask2_LPF[centerRow2-centerRectangle:centerRow2+centerRectangle, centerCol2-centerRectangle:centerCol2+centerRectangle]=1

	img1DFT = cv2.dft(np.float32(img[k+2]), flags=cv2.DFT_COMPLEX_OUTPUT)
	img2DFT = cv2.dft(np.float32(img[k+1]), flags=cv2.DFT_COMPLEX_OUTPUT)

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

	return ang1, ang2, img1DFT, img2DFT, DFTShift1, DFTShift2, img1_magnitudeSpectrum, img2_magnitudeSpectrum, overlayFFT_mag, img1_LPF_iFFT_mag,img2_LPF_iFFT_mag, img1_HPF_iFFT_mag, img2_HPF_iFFT_mag, overlayFFT_phase, CartDiff, overlayFFT,overlayiFFT, img2Corrected, img2PhaseCorrected;



OverLayFFT(img[k+2],img[k+1])
# print(DFTShift1)

overLapCorrCoefFFT= np.corrcoef(ang1[1:(row-1),(col-1):1:-1],
 ang2[1:(row2-1),1:(col2-1)], rowvar=False)[:,(col-1)]
overLapCorrCoefLPF= np.corrcoef(img1_LPF_iFFT_mag[1:(row-1),(col-1):1:-1],
	img2_LPF_iFFT_mag[1:(row2-1),1:(col2-1)], rowvar=False)[:,(col-1)]
overLapCorrCoefHPF= np.corrcoef(img1_HPF_iFFT_mag[1:(row-1),(col-1):1:-1], img2_HPF_iFFT_mag[1:(row2-1),1:(col2-1)], rowvar=False)[:,(col-1)]

# print(overLapCorrCoefAvg)

plt.figure("Spatial domain")
plt.subplot(321),plt.imshow(img[k+2],cmap = 'gray'),plt.title('Image 1')
plt.xticks(),plt.yticks([])
plt.subplot(323),plt.imshow(img[k+1],cmap = 'gray'), plt.title('Image 2')
plt.xticks(),plt.yticks([])
plt.subplot(325),plt.plot(overLapCorrCoef)
plt.title('Horizontal Corr Coef'), plt.xticks(),plt.yticks(np.arange(-0.8,1.3,0.2))

plt.subplot(322),plt.imshow(img2_translated,cmap = 'gray'),plt.title('Image 2 Translated')
plt.xticks(),plt.yticks([])
plt.subplot(324),plt.imshow(img[k+1],cmap = 'gray'), plt.title('Image 2 original')
plt.xticks(),plt.yticks([])
plt.subplot(326),plt.plot(overLapCorrCoefVertical)
plt.title('Vertical Corr Coef'), plt.xticks(),plt.yticks(np.arange(-0.8,1.3,0.2))

# plt.title('Corr Coef'), plt.xticks(),plt.yticks(np.arange(-0.8,1.3,0.2))
# plt.subplot(322),plt.imshow(img1_Avg,cmap = 'gray'),plt.title('Image 1 50 pixel Avg')
# plt.xticks(),plt.yticks([])
# plt.subplot(324),plt.imshow(img2_Avg,cmap = 'gray'), plt.title('Image 2 50 pixel Avg')
# plt.xticks(),plt.yticks([])
# plt.subplot(326),plt.plot(overLapCorrCoefAvg)
# plt.title('Avg Corr Coef'), plt.xticks(),plt.yticks(np.arange(-0.8,1.3,0.2))
plt.show()

# plt.figure("Freq domain")
# plt.subplot(331),plt.imshow(overlayFFT_phase)
# plt.title('img[2]-img[i] FFT phase difference spectrum'), plt.xticks(),plt.yticks([])
# plt.subplot(334),plt.imshow(img2Corrected,cmap = 'gray')
# plt.title('img[2] phase corrected'), plt.xticks(),plt.yticks([])
# plt.subplot(337),plt.plot(overLapCorrCoefFFT), plt.title('')
# plt.xticks(), plt.yticks(np.arange(-0.8,1.3,0.2))

# plt.subplot(332),plt.imshow(img1_LPF_iFFT_mag,cmap = 'gray')
# plt.title('img[1] iFFT LPF'), plt.xticks(),plt.yticks([])
# plt.subplot(335),plt.imshow(img2_LPF_iFFT_mag,cmap = 'gray')
# plt.title('img[2] iFFT LPF'), plt.xticks(),plt.yticks([])
# plt.subplot(338),plt.plot(overLapCorrCoefLPF), plt.title('LPF Corr Coef'),
# plt.xticks(), plt.yticks(np.arange(-0.8,1.3,0.2))

# plt.subplot(333),plt.imshow(img1_HPF_iFFT_mag)
# plt.title('img[1] iFFT HPF'), plt.xticks(),plt.yticks([])
# plt.subplot(336),plt.imshow(img2_HPF_iFFT_mag)
# plt.title('img[2] iFFT HPF'), plt.xticks(),plt.yticks([])
# plt.subplot(339),plt.plot(overLapCorrCoefHPF), plt.title('HPF Corr Coef'),
# plt.xticks(), plt.yticks(np.arange(-0.8,1.3,0.2))
# plt.show()