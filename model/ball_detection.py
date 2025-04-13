import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

# load video
video_path = os.path.join(os.getcwd(), "..", "videos", "videoplayback.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error opening video stream or file: {video_path}")
# get properties
fps = cap.get(cv2.CAP_PROP_FPS)

size = (1280, 720)


def preprocessing(frame1, frame2, frame3):
    # frame 3 is the current frame
    # frame 2 is the previous frame
    # frame 1 is the frame before the previous frame

    # size
    frame1 = cv2.resize(frame1, size)
    frame2 = cv2.resize(frame2, size)
    frame3 = cv2.resize(frame3, size)

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
    _, threshold = cv2.threshold(combined, 0, 255, cv2.THRESH_OTSU)

    # apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    processed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=5)
    processed = cv2.morphologyEx(processed, cv2.MORPH_DILATE, kernel, iterations=2)

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
    center: (float, float)  # in pixel coordinates
    radius: float

    def __init__(self, cnt):
        self.contour = cnt
        self.area = cv2.contourArea(cnt)
        # self.center = contour_center(cnt)
        self.center, self.radius = cv2.minEnclosingCircle(cnt)

    def dist_to(self, other):
        return math.sqrt((self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2)


# from sachdeva 2019
def detection_candidates_from_size(blobs: List[Blob], min_ball_area=1, max_ball_area=200, min_player_area=400) -> Tuple[
    List[Blob], List[Blob], List[Blob]]:
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
        elif c.area < min_player_area and c.area > max_ball_area:
            incomplete_player_candidates.append(c)
        elif c.area < max_ball_area and c.area > min_ball_area:
            ball_candidates.append(c)

    return ball_candidates, player_candidates, incomplete_player_candidates


# from sachdeva 2019
def detection_ball_candidates_from_player_proximity(ball_candidates: List[Blob], player_candidates: List[Blob],
                                                    incomplete_player_candidates: List[Blob],
                                                    min_ball_distance: int = 20) -> List[Blob]:
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
def detect_ball_candidates_from_motion(ball_candidates: List[Blob], ball_candidates_previous: List[Blob],
                                       min_motion_distance=0.01, max_motion_distance=100) -> List[Blob]:
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


i = 0
frame_c, frame_p, frame_pp = None, None, None
ball_candidates_prev = []

while cap.isOpened():
    # load frame
    ret, frame = cap.read()
    if not ret:
        break
    i += 1
    # assign
    frame_pp = frame_p
    frame_p = frame_c
    frame_c = frame.copy()
    if i < 3:
        continue

    # preprocess
    processed = preprocessing(frame_pp, frame_p, frame_c)

    # find contours (https://stackoverflow.com/questions/8830619/difference-between-cv-retr-list-cv-retr-tree-cv-retr-external)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the smallest blob (ball)
    if contours:
        blobs = [Blob(c) for c in contours]

        ball_candidates, player_candidates, incomplete_player_candidates = detection_candidates_from_size(blobs)
        ball_candidates = detection_ball_candidates_from_player_proximity(ball_candidates, player_candidates,
                                                                          incomplete_player_candidates)
        ball_candidates = detect_ball_candidates_from_motion(ball_candidates, ball_candidates_prev)

        print(f"Contours found: {len(contours)}")
        cv2.drawContours(frame, list(map(lambda x: x.contour, ball_candidates)), -1, (0, 0, 255), 1)

    else:
        print("No contours found")

    # draw stuff
    cv2.imshow("Frame", frame)
    cv2.imshow("After PreProcessing", processed)
    if cv2.waitKey(1000 // int(fps)) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('n'):  # skip frames manually
        continue

cap.release()
cv2.destroyAllWindows()
