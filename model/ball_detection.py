import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

# print(cv2.getBuildInformation())
cv2.ocl.setUseOpenCL(True)
print("OpenCL enabled:", cv2.ocl.useOpenCL())

# load video
video_path = os.path.join(os.getcwd(), "..", "videos", "casse_lukas_1_cut.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error opening video stream or file: {video_path}")
# get properties
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"source FPS: {fps}")

# required to detect ball at the front wall!
size = (1280, 720)


def preprocessing(frame1, frame2, frame3):
    # frame 3 is the current frame
    # frame 2 is the previous frame
    # frame 1 is the frame before the previous frame

    # convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur
    blur_size = (7, 7)
    blur1 = cv2.GaussianBlur(gray1, blur_size, 0)
    blur2 = cv2.GaussianBlur(gray2, blur_size, 0)
    blur3 = cv2.GaussianBlur(gray3, blur_size, 0)

    # subtract the twoframes
    diff1 = cv2.absdiff(blur1, blur2)
    diff2 = cv2.absdiff(blur2, blur3)

    # combine
    combined = cv2.bitwise_and(diff1, diff2)

    # apply otsu thresholding
    _, threshold = cv2.threshold(combined, 10, 255, cv2.THRESH_OTSU)

    # apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    processed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=8)

    return processed


def contour_center(contour) -> (int, int):
    """Return the center of the contour"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    return x, y


@dataclass
class Blob:
    contour: np.ndarray
    area: float  # in pixel coordinates
    center: Tuple[float, float]  # in pixel coordinates
    radius: float

    def __init__(self, cnt):
        self.contour = cnt
        self.area = cv2.contourArea(cnt)
        # self.center = contour_center(cnt)
        self.center, self.radius = cv2.minEnclosingCircle(cnt)

    def dist_to(self, other):
        return math.sqrt((self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2)


# from sachdeva 2019
def detection_candidates_from_size(blobs: List[Blob], min_ball_area=1, max_ball_area=100, min_player_area=500) -> \
        Tuple[List[Blob], List[Blob], List[Blob]]:
    """
    :param blobs: list of contours
    :param min_ball_area: minimum area of the ball
    :param min_player_area: minimum area of the player
    :param max_ball_area: maximum area of the ball
    :return: [[ball_candidates], [player_candidates], [incomplete_player_candidates]]
    """
    ball_candidates = []
    player_candidates = []
    incomplete_player_candidates = []
    for c in blobs:
        if c.area > min_player_area:
            player_candidates.append(c)
        elif min_player_area > c.area > max_ball_area:
            incomplete_player_candidates.append(c)
        elif max_ball_area > c.area > min_ball_area:
            ball_candidates.append(c)

    return ball_candidates, player_candidates, incomplete_player_candidates


# from sachdeva 2019
def detection_ball_candidates_from_player_proximity(ball_candidates: List[Blob], player_candidates: List[Blob],
                                                    incomplete_player_candidates: List[Blob],
                                                    min_ball_distance: int = 50) -> List[Blob]:
    """
    :param min_ball_distance: minimum distance between ball and player
    :param incomplete_player_candidates: list of incomplete player candidates
    :param ball_candidates: list of ball candidates
    :param player_candidates: list of player candidates
    :return: ball candidates filtered
    """
    ball_candidates_filtered = []
    if len(ball_candidates) == 0:
        return ball_candidates_filtered

    min_dist = 8000000  # large number

    for c in ball_candidates:
        if len(player_candidates) > 1:
            for p in player_candidates:
                dist = c.dist_to(p)
                if dist < min_dist:
                    min_dist = dist
        elif len(player_candidates) == 1:
            p = player_candidates[0]
            dist = c.dist_to(p)
            if dist < min_dist:
                min_dist = dist
            for pp in incomplete_player_candidates:
                dist = c.dist_to(pp)
                if dist < min_dist:
                    min_dist = dist
        elif len(incomplete_player_candidates) > 1:
            for pp in incomplete_player_candidates:
                dist = c.dist_to(pp)
                if dist < min_dist:
                    min_dist = dist

        if min_dist > min_ball_distance:
            ball_candidates_filtered.append(c)

    return ball_candidates_filtered


# from sachdeva 2019
def detection_ball_candidates_from_motion(ball_candidates: List[Blob], ball_candidates_previous: List[Blob],
                                          min_motion_distance=10, max_motion_distance=100) -> List[Blob]:
    """
    :param ball_candidates:
    :param ball_candidates_previous:
    :param min_motion_distance:
    :param max_motion_distance:
    :return: filtered ball candidates
    """
    ball_candidates_filtered = []
    if len(ball_candidates) == 0:
        return ball_candidates_filtered
    if len(ball_candidates_previous) == 0:
        return ball_candidates
    for c in ball_candidates:
        flag = False
        for pc in ball_candidates_previous:
            dist = c.dist_to(pc)
            if min_motion_distance < dist < max_motion_distance:
                flag = True
                break
        if flag:
            ball_candidates_filtered.append(c)
    return ball_candidates_filtered


class BallTrackerKalman:
    position: Tuple[float, float] = (size[0] // 2, size[1] // 2)  # in pixel coordinates

    def __init__(self):
        self._initialized = False

        # set up kalman
        self.kalman = cv2.KalmanFilter(4, 2)

    def _initialize(self, ball_candidates: List[Blob]):
        image_center = (size[0] // 2, size[1] // 2)
        if len(ball_candidates) == 0:
            # use center of image
            self.position = image_center
        elif len(ball_candidates) == 1:
            # use the only candidate
            self.position = ball_candidates[0].center
        else:
            # use the one closest to center
            best_candidate = None
            best_dist = None
            for c in ball_candidates:
                dist = math.sqrt((c.center[0] - image_center[0]) ** 2 + (c.center[1] - image_center[1]) ** 2)
                if best_dist is None or dist < best_dist:
                    best_candidate = c
                    best_dist = dist
            self.position = best_candidate.center

    def _update(self, ball_candidates: List[Blob]):
        if len(ball_candidates) == 0:
            # only predict using kalman filter
            prediction = self.kalman.predict().flatten()
            self.position = (prediction[0], prediction[1])
        elif len(ball_candidates) == 1:
            # the location is assumed to be the real ball location. used to correct the Kalman filter
            self.position = ball_candidates[0].center
            self.kalman.correct(np.array([[np.float32(self.position[0])], [np.float32(self.position[1])]]))
        else:  # multiple candidates
            # use the one closest to the prediction and correct the Kalman filter
            prediction = self.kalman.predict().flatten()
            best_candidate = None
            best_dist = None
            for c in ball_candidates:
                dist = math.sqrt((c.center[0] - prediction[0]) ** 2 + (c.center[1] - prediction[1]) ** 2)
                if best_dist is None or dist < best_dist:
                    best_candidate = c
                    best_dist = dist
            self.position = best_candidate.center
            self.kalman.correct(np.array([[np.float32(self.position[0])], [np.float32(self.position[1])]]))

    def track(self, ball_candidates: List[Blob]) -> None:
        if self._initialized:
            self._update(ball_candidates)
        else:
            self._initialize(ball_candidates)
            self._initialized = True


class BallTrackerHolt:
    """
    using holt's double exponential smoothing
    """
    level = np.array([np.float32(size[0] // 2), np.float32(size[1] // 2)])
    trend: np.ndarray = np.zeros((2, 1))  # in pixel coordinates

    def __init__(self, alpha: float = 0.9, beta: float = 0.1):
        """
        :param alpha: (level smoothing) controls how closely you follow recent observations (higher αα → more weight on new data).
        :param beta: (trend smoothing) controls how quickly you change the trend (higher ββ → shorter trend memory).
        """
        self._initialized = False
        self.alpha = alpha
        self.beta = beta

    def _predict(self):
        return self.level + self.trend

    def _update(self):
        # Holt's double exponential smoothing
        self.level = self.alpha * self.level + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - self.level) + (1 - self.beta) * self.trend

    def track(self, ball_candidates: List[Blob]) -> None:
        if len(ball_candidates) == 0:
            # only predict
            self.level = self._predict()
            # maybe update the level based on the prediction?
        elif len(ball_candidates) == 1:
            pass
        else:
            # use the one closest to the prediction and correct the Kalman filter
            prediction = self._predict()
            best_candidate = None
            best_dist = None
            for c in ball_candidates:
                dist = math.sqrt((c.center[0] - prediction[0]) ** 2 + (c.center[1] - prediction[1]) ** 2)
                if best_dist is None or dist < best_dist:
                    best_candidate = c
                    best_dist = dist
            self.level = best_candidate.center

    @property
    def position(self) -> Tuple[float, float]:
        return self.level.flatten().tolist()


# global state
i = 0
frame_c, frame_p, frame_pp = None, None, None
ball_candidates_prev = []
tracker = BallTrackerKalman()

# main work loop
while cap.isOpened():
    # load frame
    ret, frame = cap.read()
    if not ret:
        break
    i += 1

    # fix size
    frame = cv2.resize(frame, size)

    # move to gpu
    frame_gpu = cv2.UMat(frame)

    # assign
    frame_pp = frame_p
    frame_p = frame_c
    frame_c = frame_gpu
    if i < 3:
        continue

    # preprocess
    processed = preprocessing(frame_pp, frame_p, frame_c)

    # find contours (https://stackoverflow.com/questions/8830619/difference-between-cv-retr-list-cv-retr-tree-cv-retr-external)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the smallest blob (ball)
    if contours:
        # convert to blobs
        blobs = [Blob(c) for c in contours]

        # get ball and player candidates
        ball_candidates_size, player_candidates, incomplete_player_candidates = detection_candidates_from_size(blobs)
        ball_candidates_proximity = detection_ball_candidates_from_player_proximity(ball_candidates_size,
                                                                                    player_candidates,
                                                                                    incomplete_player_candidates)
        ball_candidates_motion = detection_ball_candidates_from_motion(ball_candidates_proximity, ball_candidates_prev)
        print(f"Ball candidates: {len(ball_candidates_motion)}")
        # update previous ball candidates
        ball_candidates_prev = ball_candidates_motion  # FIXME: or update it later with the single result of the tracker?
        # visualize candidates
        # cv2.drawContours(frame, list(map(lambda x: x.contour, ball_candidates_motion)), -1, (0, 0, 255), 30)  # FIXME: does not work
        for c in ball_candidates_motion:
            cv2.circle(frame, (int(c.center[0]), int(c.center[1])), int(c.radius) + 2, (0, 0, 255), 3)

        # player tracking narrows it down to a single player candidate
        tracker.track(ball_candidates_motion)
        # draw the corrected position
        cv2.circle(frame, (int(tracker.position[0]), int(tracker.position[1])), 20, (0, 255, 0), 3)

    else:
        print("No contours found")

    # draw stuff
    cv2.imshow("Frame", frame)
    cv2.imshow("After PreProcessing", processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
