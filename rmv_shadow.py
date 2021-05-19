import cv2
import numpy as np

img = cv2.imread('april_white.jpg')
img = cv2.resize(img, (720, 720))
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
cv2.imshow('Luminance', l)

ret2, th = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
mask1 = cv2.bitwise_and(img, img, mask = th)
cv2.imshow('mask1', mask1)
cv2.waitKey(0)