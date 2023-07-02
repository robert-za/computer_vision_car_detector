import cv2 as cv

img = cv.imread("Resources/Photos/cats.jpg")
cv.imshow("Cats", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

# Simple thresholding
threshold, thresh = cv.threshold(gray, 200, 150, cv.THRESH_BINARY)
cv.imshow("Simple Thresh", thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow("Simple Thresh Inversed", thresh_inv)

# Adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 9)
cv.imshow("Adaptive Threshold", adaptive_thresh)


cv.waitKey(0)