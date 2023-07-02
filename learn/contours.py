import cv2 as cv
import numpy as np

img = cv.imread("Resources/Photos/cats.jpg")
cv.imshow("Cats", img)

blank = np.zeros(img.shape, dtype="uint8")
cv.imshow("Blank", blank)

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Grey", grey)

blurred = cv.GaussianBlur(grey, (5,5), cv.BORDER_DEFAULT)
cv.imshow("Blurred", blurred)

canny = cv.Canny(blurred, 125, 175)
cv.imshow("Canny Edges", canny)

ret, thresh = cv.threshold(grey, 125, 255, cv.THRESH_BINARY)
cv.imshow("Thresh", thresh)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

print(len(contours))

cv.drawContours(blank, contours, -1, (0,0,255), thickness=1)
cv.imshow("Countours", blank)

cv.waitKey(0)