import numpy as np
import cv2 as cv
import os

BASE_PATH = r"/home/robert/Workspace/opencv-livefeed/learn/Resources/Faces/train"
people = [folder for folder in os.listdir(BASE_PATH)]


haar_cascade = cv.CascadeClassifier("haar_face.xml")

features = np.load("features.npy", allow_pickle=True)
labels = np.load("labels.npy", allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'/home/robert/Workspace/opencv-livefeed/learn/Resources/Faces/val/ben_affleck/2.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person", gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f"Label = {people[label]} with a confidence of {confidence}.")
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2)

cv.imshow("Detected Face", img)

cv.waitKey(0)