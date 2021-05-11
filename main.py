from utils import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

from utils import findExternalContour3, getAprilTags, resizeTo, \
    getMaxContour, crop, order_points, findExternalContour2, getCannyEdges, \
    four_point_transform, projectToContour, getBoundingRectFromLabelStat, \
    getBoundingRectCenterFromLabelStat, getFilteredContours, getFilteredLabelIndex, \
    correctPerspectiveWithAprilTags, floodFillCustom, getSortedContours, \
    findExternalContour, applyAspecRatioToContour
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.resize(image, (1080, 1080))
h,w,chn = image.shape
ratio = image.shape[0] / 1080.0
orig = image.copy()
#image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
cv2.imshow("Scanned", warped)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped1 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped1, 11, offset = 10, method = "gaussian")
warped1 = (warped1 > T).astype("uint8") * 255
# # show the original and scanned images
print("STEP 3: Apply perspective transform")
# cv2.imshow("Original", orig)
# cv2.imshow("Scanned", warped)

seed = (70, 70)

foreground, birdEye = floodFillCustom(warped, seed)
cv2.circle(birdEye, seed, 50, (0, 255, 0), -1)
cv2.imshow("originalImg", birdEye)

cv2.circle(birdEye, seed, 100, (0, 255, 0), -1)

cv2.imshow("foreground", foreground)
cv2.imshow("birdEye", birdEye)

gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

threshImg = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
h_threshold,w_threshold = threshImg.shape
area = h_threshold*w_threshold
print(area/2)

cv2.imshow("threshImg", threshImg)

cv2.waitKey(0)

# find distinct objects
(numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
    threshImg, 4, cv2.CV_32S)

filteredIdx = getFilteredLabelIndex(stats, areaHighLimit=area/2) # here we have to ensure that the height and the weight of the rectangle is neither to big or too small.

for i in filteredIdx:
    # extract the connected component statistics for the current label
    rect = getBoundingRectFromLabelStat(stats[i])
    cv2.rectangle(birdEye, rect[0], rect[1], (255, 255, 255), 3)
  
    componentMask = (labels == i).astype("uint8") * 255
    cv2.waitKey(0)

    cv2.imshow("componentMask", componentMask)
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    componentMask = cv2.dilate(componentMask, kernel, iterations=3)

    cntrs = getMaxContour(componentMask)
    cv2.drawContours(birdEye, [cntrs], -1, (255, 0, 255), 8)
    cv2.imshow("contour", birdEye)

cv2.imshow("original contour", birdEye)
cv2.waitKey(0)
cv2.destroyAllWindows()