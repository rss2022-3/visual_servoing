import cv2
import numpy as np
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########
	#low_orange = np.array([70, 200, 200])
	low_orange = np.array([70, 200, 200])
	high_orange = np.array([130, 255, 255])
	hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	mask = cv2.inRange(hsv_image, low_orange, high_orange)
	new_image = cv2.bitwise_and(img,img, mask=mask)
	new_image_rgb = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
	gray_image = cv2.cvtColor(new_image_rgb, cv2.COLOR_RGB2GRAY)
	ret, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
	erode_img = cv2.erode(thresh, np.ones((5,5), 'uint8'), iterations=2)
	dilate_img = cv2.dilate(erode_img,np.ones((5,5), 'uint8'), iterations=2)
	_, contours, hierarchy = cv2.findContours(image=dilate_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
	img_copy = img.copy()
	cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1,color=(0,255,0),thickness=2,lineType=cv2.LINE_AA)
	cnt = contours[0]
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(img_copy, (x,y), (x+w, y+h), (255,0,0),2)
	bounding_box = ((x,y),(x+w,y+h))
	# Return bounding box
	return bounding_box
