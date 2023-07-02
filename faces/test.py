import cv2 as cv

webcam = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier("haar_face.xml")
stop=False

while not stop:
    ret, frame = webcam.read()
    
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        for (x,y,w,h) in faces_rect:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv.imshow("Detected Faces", frame)

        # cv.imshow("Robert", gray)
        key = cv.waitKey(1)
        if key == ord("q"):
            break

webcam.release()
cv.destroyAllWindows()
