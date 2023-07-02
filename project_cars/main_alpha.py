import cv2 as cv
import numpy as np

cap = cv.VideoCapture('cars.mp4')
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

background_img = cv.imread("frame.jpg")
background_img = cv.cvtColor(background_img, cv.COLOR_BGR2GRAY)
background_img = cv.resize(background_img, (frame_width, frame_height))
background_img = cv.convertScaleAbs(background_img)

class CarCounter:
    def __init__(self,
                 contours_number=10,
                 min_area = 170):
        self.contours_number = contours_number
        self.min_area = min_area
        self.font = cv.FONT_HERSHEY_SIMPLEX

        self.prev_centroids = []

    def main_loop(self, background_img):

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # frame_id = int(cap.get(1))
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                blank = np.zeros(background_img.shape[:2], dtype="uint8")
                rectangle = cv.rectangle(blank.copy(), (250,170), (450, 300), 255, -1)

                background_img = cv.convertScaleAbs(background_img)
                subtracted_img = cv.absdiff(background_img, gray)
                _, threshold_img = cv.threshold(subtracted_img, 40, 255, 0)
                cv.imshow("SHOW", threshold_img)
                # masked = cv.bitwise_and(threshold_img, threshold_img, mask=rectangle)
                # self.bind_objects(gray, threshold_img)
                # color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
                # cv.imshow('Frame', color)

                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        
        cap.release()
        cv.destroyAllWindows()

        cv.waitKey(0)

    def draw_bounding_boxes(self,frame,contour_id,bounding_points,cx,cy,prev_cx,prev_cy):
        cv.drawContours(frame,[bounding_points],0,(0,255,0),1)
        cv.line(frame,(prev_cx,prev_cy),(cx,cy),(0,0,255),1)
        cv.circle(frame,(cx,cy),3,(0,0,255),4)
        cv.putText(frame,str(contour_id),(cx,cy-15),self.font,0.4,(255,0,0),2)

    def bind_objects(self,frame,thresh_img):
        cnts,_ = cv.findContours(thresh_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts,key = cv.contourArea,reverse=True)[:self.contours_number]

        cnt_id         = 1
        cur_centroids  = []
        for c in cnts:
            if cv.contourArea(c) < self.min_area:
                continue
            rect   = cv.minAreaRect(c)
            points = cv.boxPoints(rect)
            points = np.int0(points)

            cx = int(rect[0][0])
            cy = int(rect[0][1])

            w,h = rect[1]

            C = np.array((cx,cy))
            cur_centroids.append((cx,cy))

            if len(self.prev_centroids) == 0: 
                prev_cx,prev_cy = cx,cy
            elif len(cnts) == 0: 
                prev_cx,prev_cy = cx,cy
            else:
                minPoint = None
                minDist = None
                for i in range(len(self.prev_centroids)):
                    dist = np.linalg.norm(C - self.prev_centroids[i])
                    if (minDist is None) or (dist < minDist):
                        minDist = dist
                        minPoint = self.prev_centroids[i]
                if minDist < w/2:
                    prev_cx,prev_cy = minPoint
                else: 
                    prev_cx,prev_cy = cx,cy
            
            self._draw_bounding_boxes(frame,cnt_id,points,cx,cy,prev_cx,prev_cy)

            cnt_id += 1
        self.prev_centroids = cur_centroids

cars_counter = CarCounter()
cars_counter.main_loop(background_img)
