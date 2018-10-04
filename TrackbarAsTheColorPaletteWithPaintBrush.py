import cv2
import numpy as np 

drawing = False # true if mouse is pressed
ix,iy = -1,-1

# mouse callback function
# def draw_circle(event,x,y,flags, param):
# 	global ix,iy,drawing

# 	if event==cv2.EVENT_LBUTTONDOWN:
# 		drawing = True
# 		ix,iy = x,y

# 	elif event == cv2.EVENT_MOUSEMOVE:
# 		if drawing == True:
# 			cv2.circle(img,(x,y),10,(255,0,0),-1)

# 	elif event == cv2.EVENT_LBUTTONUP:
# 		drawing = False
# 		cv2.circle(img,(x,y),10,(255,0,0),-1)

def draw_circle(event,x,y,flags, param):
	if event==cv2.EVENT_LBUTTONDBLCLK:
		cv2.circle(img,(x,y),100,(255,0,0),-1)


def nothing(x):
	pass


# create a black image in a window and bind the function to the window
img = np.zeros((300,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)


# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch,'image',0,1,nothing)


while(1):
	cv2.imshow('image',img)
	k = cv2.waitKey(1) & 0xFF # 0xFF is hex for integer 255
	if k == 32:
		break

	# get current positions of four trackbars
	r = cv2.getTrackbarPos('R','image')
	g = cv2.getTrackbarPos('G','image')
	b = cv2.getTrackbarPos('B','image')
	s = cv2.getTrackbarPos(switch, 'image')

	if s == 0:
		img[:] = 0
	else:
		img[:] = [b,g,r]



cv2.destroyAllWindows()
