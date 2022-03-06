import cv2
import imutils
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
  Helper function to print out images, for debugging.
  Press any key to continue.
  """
  winname = "Image"
  cv2.namedWindow(winname)        # Create a named window
  cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
  cv2.imshow(winname, img)
  cv2.waitKey()
  cv2.destroyAllWindows()

def cd_sift_ransac(img, template):
  """
  Implement the cone detection using SIFT + RANSAC algorithm
  Input:
    img: np.3darray; the input image with a cone to be detected
  Return:
    bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
        (x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
  """
  # Minimum number of matching features
  MIN_MATCH = 10
  # Create SIFT
  sift = cv2.xfeatures2d.SIFT_create()

  # Compute SIFT on template and test image
  kp1, des1 = sift.detectAndCompute(template,None)
  kp2, des2 = sift.detectAndCompute(img,None)

  # Find matches
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2,k=2)

  # Find and store good matches
  good = []
  for m,n in matches:
    if m.distance < 0.75*n.distance:
      good.append(m)

  # If enough good matches, find bounding box
  if len(good) > MIN_MATCH:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # Create mask
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = template.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    #print(dst)
    #print("just printed dst")
    x_min = y_min = x_max = y_max = 0
    img2 = cv2.polylines(img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    #print(img2)
    image_print(img2)
    # Return bounding box
    #print(dst)
    #print(dst[0][0][0], dst[1][0][1])
    x_min= min(dst[0][0][0], dst[1][0][0], dst[2][0][0], dst[3][0][0])
    y_min = min(dst[0][0][1], dst[1][0][1], dst[2][0][1], dst[3][0][1])
    x_max= max(dst[0][0][0], dst[1][0][0], dst[2][0][0], dst[3][0][0])
    y_max = max(dst[0][0][1], dst[1][0][1], dst[2][0][1], dst[3][0][1])
    return ((x_min, y_min),(x_max, y_max))
  else:

    print "[SIFT] not enough matches; matches: ", len(good)

    # Return bounding box of area 0 if no match found
    return ((0,0), (0,0))

def cd_template_matching(img, template):
  """
  Implement the cone detection using template matching algorithm
  Input:
    img: np.3darray; the input image with a cone to be detected
  Return:
    bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
        (x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
  """
  template_canny = cv2.Canny(template, 50, 200)

  # Perform Canny Edge detection on test image
  grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_canny = cv2.Canny(grey_img, 50, 200)

  # Get dimensions of template
  (img_height, img_width) = img_canny.shape[:2]

  # Keep track of best-fit match
  best_match = None
  best_max_val = None
  best_max_loc = 0
  best_hw = (0,0)
  best_r = 0

  # Loop over different scales of image
  for scale in np.linspace(1.5, .5, 50):
    # Resize the image
    resized_template = imutils.resize(template_canny, width = int(template_canny.shape[1] * scale))
    r = template_canny.shape[1]/float(resized_template.shape[1])
    (h,w) = resized_template.shape[:2]
    # Check to see if test image is now smaller than template image
    if resized_template.shape[0] > img_height or resized_template.shape[1] > img_width:
      continue

    ########## YOUR CODE STARTS HERE ##########
    # Use OpenCV template matching functions to find the best match
    # across template scales.
    res = cv2.matchTemplate(img_canny, resized_template, cv2.TM_CCOEFF)
    #res = cv2.matchTemplate(grey_img, resized_template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if best_max_val is None or max_val > best_max_val:
      best_max_val = max_val
      best_match = res
      best_max_loc = max_loc
      best_hw = (h,w)
      best_r = 0
    # Remember to resize the bounding box using the highest scoring scale
    # x1,y1 pixel will be accurate, but x2,y2 needs to be correctly scaled
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(best_match)
#  #print(max_loc)
  top_left = max_loc
  bottom_right = (top_left[0] + best_hw[1], top_left[1] + best_hw[0])
#  #print(max_loc, bottom_right)
  #loc = np.where(best_match >= .8)
  #print("just before")
  cv2.rectangle(img, top_left, bottom_right,255,2),
  #for pt in zip(*loc[::-1]):
    #print("in here")
    #print(pt[0] + best_hw[1], pt[1] + best_hw[0])
    #cv2.rectangle(img, pt, (pt[0] + best_hw[1], pt[1] + best_hw[0]),(0,255,255),2)
  #cv2.rectangle(img,min_loc,bottom_right,255,5)
  bounding_box = (top_left, bottom_right)
    #bounding_box = ((0,0),(0,0))
    ########### YOUR CODE ENDS HERE ########### 
  #(startX, startY) = (int(best_max_loc[0]*r), int(best_max_loc[1]*r))
  #(endX, endY) = (int((best_max_loc[0] + best_hw[1])*r),int((best_max_loc[1]+best_hw[0])*r))
  #cv2.rectangle(img,(startX,startY),(endX,endY),(0,0,255),2)
  #bounding_box = ((startX, startY),(endX, endY))
  print(bounding_box)
  image_print(img)
  

  image_print(img)
  return bounding_box
