import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype="uint8")

# cv.imshow("Blank", blank)

# 1. Paint image a certain color
blank[200:300, 300:400] = 0, 0, 255
# cv.imshow("Green", blank)

# 2. Draw a rectangle
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=2)
# cv.imshow("Rectangle", blank)

# 3. Draw circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=1)
# cv.imshow("Circle", blank)

# 4. Draw line
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255, 255, 255), thickness=5)
# cv.imshow("Line", blank)

# 5. Write text on image
cv.putText(blank, "Hello, World!", (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), thickness=2)
cv.imshow("Text", blank)

cv.waitKey(0)