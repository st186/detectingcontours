import numpy as np
import cv2
import imutils


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def floodFillCustom(originalImage, seed):
    originalImage = np.maximum(originalImage, 10)
    foreground = originalImage.copy()

    # Use the top left corner as a "background" seed color (assume pixel [10,10] is not in an object).
    cv2.circle(originalImage, seed, 50, (0, 255, 0), -1)

    # Use floodFill for filling the background with black color
    cv2.floodFill(foreground, None, seed, (0, 0, 0),
                  loDiff=(5, 5, 5), upDiff=(5, 5, 5))
    return [foreground, originalImage]
    # h,w,chn = originalImage.shape

    # mask = np.zeros((h+2,w+2),np.uint8)

    # floodflags = 4
    # floodflags |= cv2.FLOODFILL_MASK_ONLY
    # floodflags |= (255 << 8)

    # num,im,mask,rect = cv2.floodFill(originalImage, mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)

    # return [mask, originalImage]

# Convert to Grayscale


def correctPerspectiveWithAprilTags(originalImg, margin=15, printTagOnImage=True):
    tags = getAprilTags(originalImg, printTagOnImage)

    sortedTags = sorted(list(zip(tags[0], tags[1])), key=lambda x: x[1])

    # if len(tags[0]) == 4: //2,3,0,1
    tag1Coordinates = sortedTags[0][0][0][0]
    tag2Coordinates = sortedTags[1][0][0][1]
    tag3Coordinates = sortedTags[2][0][0][2]
    tag4Coordinates = sortedTags[3][0][0][3]

    pts = np.array([tag1Coordinates+[margin, 0], tag2Coordinates + [margin, 0],
                    tag3Coordinates+[-margin, 0], tag4Coordinates + [-margin, 0]]).reshape(4, 2)

    return four_point_transform(originalImg, pts)
# else:
    # return originalImg


def crop(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y+h, x:x+w]


def getCannyEdges(image, isBW=False) -> np.ndarray:
    if isBW == False:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 30, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged


def projectToContour(originalImg, resizeToFit=False):
    # to reduce computation time for edge detection, we resize the image to a smaller version
    smallScaledImg = imutils.resize(originalImg, height=500)
    ratio = originalImg.shape[0] / 500.0

    # convert the image to grayscale, blur it, and find edges
    # in the image
    edgedImg = getCannyEdges(smallScaledImg)
    edgedImgCnt = findExternalContour(edgedImg)
    pts = edgedImgCnt.reshape(4, 2) * ratio
    birdEye = four_point_transform(originalImg, pts)
    if resizeToFit:
        imutils.resize(birdEye, height=originalImg.shape[0])
    return birdEye


def getFilteredContours(image, minAreaFilter=20000) -> np.array:
    ret = []
    ctrs, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ctrs = sorted(ctrs, key=cv2.contourArea, reverse=True)
    #ctrs = ctrs[:10]
    for i, c in enumerate(ctrs):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        if area < minAreaFilter:
            break
        ret.append(c)
    return ret


def getMaxContour(image) -> np.array:
    ret = []
    ctrs, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ctrs = sorted(ctrs, key=cv2.contourArea, reverse=True)
    return ctrs[0]


def getSortedContours(image) -> np.array:
    ret = []
    ctrs, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ctrs = sorted(ctrs, key=cv2.contourArea, reverse=True)
    return ctrs


def getFilteredLabelIndex(stats, areaHighLimit, widthLowLimit=10, heightLowLimit=10, areaLowLimit=1000):
    ret = []
    for i in range(1, stats.shape[0]):
        # extract the connected component statistics for the current label
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        keepWidth = w > widthLowLimit
        keepHeight = h > heightLowLimit
        keepArealow = area > areaLowLimit
        keepAreahigh = area < areaHighLimit

        if all((keepWidth, keepHeight, keepArealow, keepAreahigh)):
            ret.append(i)

    return ret


def contourOffset(cnt, offset):
    """ Offset contour, by 5px border """
    # Matrix addition
    cnt += offset
    # if value < 0 => replace it by 0
    cnt[cnt < 0] = 0
    return cnt


def findExternalContour3(edgeImg):
    contours, hierarchy = cv2.findContours(
        edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Finding contour of biggest rectangle
    # Otherwise return corners of original image
    # Don't forget on our 5px border!
    height = edgeImg.shape[0]
    width = edgeImg.shape[1]
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)
    # Page fill at least half of image, then saving max area found
    maxAreaFound = MAX_COUNTOUR_AREA * 0.5
    # Saving page contour
    pageContour = np.array(
        [[5, 5], [5, height-5], [width-5, height-5], [width-5, 5]])
    # Go through all contours
    for cnt in contours:
        # Simplify contour
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        # Page has 4 corners and it is convex
        # Page area must be bigger than maxAreaFound
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
            maxAreaFound = cv2.contourArea(approx)
            pageContour = approx


def getAprilTags(image, showOnImage=False, Tag=cv2.aruco.DICT_4X4_1000):
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        image, arucoDict, parameters=arucoParams)

    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        rects = []

        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            (topLeft, topRight, bottomRight,
             bottomLeft) = markerCorner.reshape((4, 2))
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            rects.append([topLeft, topRight, bottomRight, bottomLeft])

            if showOnImage:
                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 0, 255), 2)
                cv2.line(image, topRight, bottomRight, (0, 0, 255), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 0, 255), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 0, 255), 2)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                            (topLeft[0], topLeft[1] -
                             15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
        return (corners, ids, rejected, rects)


def resizeTo(image, width=100):
    img = image.copy()
    return imutils.resize(img, width=width)


def applyAspecRatioToContour(contours, img_orig, img_resized):
    coef_y = img_orig.shape[0] / img_resized.shape[0]
    coef_x = img_orig.shape[1] / img_resized.shape[1]

    for contour in contours:
        contour[:, :, 0] = contour[:, :, 0] * coef_x
        contour[:, :, 1] = contour[:, :,  1] * coef_y
    return contours


def findExternalContour2(image):
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # only keep contour, not hierarchy contours = list(array(contour),array(hierarchy))
    # cnts = imutils.grab_contours(cnts) same as above line
    # sort from biggest to smallest and only keep first 5 contours (0 to 4)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    return cnts[0]


def getBoundingRectFromLabelStat(stat):
    x = stat[cv2.CC_STAT_LEFT]
    y = stat[cv2.CC_STAT_TOP]
    w = stat[cv2.CC_STAT_WIDTH]
    h = stat[cv2.CC_STAT_HEIGHT]
    return[(x, y), (x + w, y + h)]


def getBoundingRectCenterFromLabelStat(stat):
    (cX, cY) = centroids[i]
    return (int(cX), int(cY))


def findExternalContour(image, originalImage=None):
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # only keep contour, not hierarchy contours = list(array(contour),array(hierarchy))
    cnts = cnts[0]
    # cnts = imutils.grab_contours(cnts) same as above line
    # sort from biggest to smallest and only keep first 5 contours (0 to 4)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)  # perimetre of the curve (=length)
        # approximation grossière ne garde que 2% de la longeur
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if not originalImage is None:
            originCopy = originalImage.copy()
            cv2.drawContours(originCopy, [approx], -1, (255, 0, 0), 2)
            cv2.imshow("approx", originCopy)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            return screenCnt
