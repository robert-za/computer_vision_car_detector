import cv2 as cv
import numpy as np

img = cv.imread("Resources/Photos/park.jpg")

cv.imshow("Boston", img)

# Translation
def translate(img, x, y):
    trans_mat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, trans_mat, dimensions)

# -x --> left
# -y --> up
# x --> right
# y --> down

translated = translate(img, -100, 100)
cv.imshow("Translated", translated)

# Rotation
def rotate(img, angle, rotation_point=None):
    height, width = img.shape[:2]

    if rotation_point is None:
        rotation_point = (width//2, height//2)

    rot_mat = cv.getRotationMatrix2D(rotation_point, angle, 1.0)
    dimensions = width, height
    return cv.warpAffine(img, rot_mat, dimensions)

rotated = rotate(img, 45)
cv.imshow("Rotated", rotated)

# Resizing
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow("Resized", resized)

# Flipping 0, 1, -1
flip = cv.flip(img, 0)
cv.imshow("Flipped", flip)

# Cropping
cropped = img[200:400, 300:400]
cv.imshow("Cropped", cropped)

cv.waitKey(0)