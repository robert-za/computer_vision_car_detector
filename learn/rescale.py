import cv2 as cv

# img = cv.imread("Resources/Photos/cat_large.jpg")
# cv.imshow("Cat", img)

def rescale_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def change_res(width, height):
    """Live video."""
    capture.set(3, width)
    capture.set(4, height)


# Reading videos
capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
    is_true, frame = capture.read()
    frame_resized = rescale_frame(frame)
    cv.imshow('Video', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('q'):
        break

capture.release()
cv.destroyAllWindows()
