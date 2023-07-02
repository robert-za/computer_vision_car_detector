import cv2 as cv

webcam = cv.VideoCapture(1)

stop=False

while not stop:
    ret, frame = webcam.read()
    
    if ret:
        # canny = cv.Canny(frame, 150, 200)
        # cv.imshow("Test123", canny)
        cv.imshow("Robert", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break

webcam.release()
cv.destroyAllWindows()
