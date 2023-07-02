import cv2 as cv
import numpy as np


class CarHandler:
    def __init__(self, video_cap):
        self.video_cap = video_cap
        self.starting_frame = 10
        self.black_mask = None
        self.average_background = None
        self.rate_of_learning = 0.01
        self.frame_width = int(video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(video_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.contours_number = 10
        self.min_area = 700
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.left_line_position = 0.2
        self.right_line_position = 0.8
        self.prev_centroids = []
        self.cars = []


    def main(self):
        while(video_cap.isOpened()):
            is_receiving, frame = video_cap.read()
            if is_receiving:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                thresholded = self.transform_gray_to_thresholded(gray)
                self.bind_objects(frame, thresholded)
                with_lines = self.set_veritcal_line(frame)
                cv.imshow("Vehicule Recognition", with_lines)
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                print("Cars moving to the left:", self.cars_moving_left)
                print("Cars moving to the right:", self.cars_moving_right)
                break
        video_cap.release()
        cv.destroyAllWindows()
        cv.waitKey(0)


    def setup_average_background(self, frame) -> None:
        if self.average_background is None:
            self.average_background = np.float32(frame)
        else:
            cv.accumulateWeighted(frame, self.average_background, self.rate_of_learning)
        return cv.convertScaleAbs(self.average_background)


    def transform_gray_to_thresholded(self, gray_frame):
        background = self.setup_average_background(gray_frame)
        subtracted = cv.absdiff(background, gray_frame)
        blurred = cv.GaussianBlur(subtracted, (21, 21), 0)
        blurred = cv.GaussianBlur(blurred, (77, 77), 0)
        dilated = cv.dilate(blurred, None)
        cv.imshow("Dilated", dilated)
        _, thresholded  = cv.threshold(dilated, 15, 255, 0)
        cv.imshow("Tresh", thresholded)
        return thresholded


    def bind_objects(self, frame, thresh_img):
        cnts,_ = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:self.contours_number]

        for c in cnts:
            if cv.contourArea(c) < self.min_area:
                continue
            rect = cv.minAreaRect(c)
            points = cv.boxPoints(rect)
            points = np.intp(points)

            cx = int(rect[0][0])
            cy = int(rect[0][1])
            w, h = rect[1]
            centroid = (cx, cy)

            car = None
            for existing_car in self.cars:
                dist = np.linalg.norm(np.array(existing_car.centroid) - np.array(centroid))
                if dist < w / 2:
                    car = existing_car
                    break

            if car is None:
                car = Car(centroid)
                print(f"{len(self.cars)} car created!")
                self.cars.append(car)
            else:
                car.update(centroid)

            self.draw_bounding_boxes(frame, points, cx, cy, car.prev_centroid[0], car.prev_centroid[1])
            self.count_events()


    def count_events(self):
        self.cars_moving_left = 0
        self.cars_moving_right = 0

        for car in self.cars:
            if car.direction == "left":
                self.cars_moving_left += 1
            elif car.direction == "right":
                self.cars_moving_right += 1


    def draw_bounding_boxes(self, frame, bounding_points, cx, cy, prev_cx, prev_cy):
        cv.drawContours(frame, [bounding_points], 0, (0, 255, 0), 1)
        cv.line(frame, (prev_cx, prev_cy),(cx, cy), (0, 0, 255), 1)
        cv.circle(frame,(cx, cy), 3, (0, 0, 255), -1)


    def set_veritcal_line(self, frame):
        left_start_point = (int(self.frame_width * self.left_line_position), 0)
        left_end_point = (int(self.frame_width * self.left_line_position), self.frame_height)
        right_start_pont = (int(self.frame_width * self.right_line_position), 0)
        right_end_pont = (int(self.frame_width * self.right_line_position), self.frame_height)
        color = (0, 255, 0)
        left_line = cv.line(frame, left_start_point, left_end_point, color, 2)
        add_right_line = cv.line(left_line, right_start_pont, right_end_pont, color, 2)
        return add_right_line


class Car:
    def __init__(self, centroid):
        self.centroid = centroid
        self.prev_centroid = centroid
        self.direction = None

    def update(self, centroid):
        self.prev_centroid = self.centroid
        self.centroid = centroid
        self.update_direction()

    def update_direction(self):
        if self.prev_centroid[0] < self.centroid[0]:
            self.direction = "right"
        else:
            self.direction = "left"


video_cap = cv.VideoCapture('cars1.mp4')
car_handler = CarHandler(video_cap)
car_handler.main()