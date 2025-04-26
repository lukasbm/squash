import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from operator import itemgetter
from typing import List, Tuple

import cv2
import numpy as np

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
size = (1024, 576)


def preprocessing(frame1, frame2, frame3):
    # frame 3 is the current frame
    # frame 2 is the previous frame
    # frame 1 is the frame before the previous frame

    # convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # FIXME: this is repeated 3 times for a frame as it passes through
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)  # FIXME: this is repeated 3 times for a frame as it passes through

    # apply Gaussian blur
    blur_size = (5, 5)  # FIXME: or 7,7?
    blur1 = cv2.GaussianBlur(gray1, blur_size, 0)
    blur2 = cv2.GaussianBlur(gray2, blur_size, 0)  # FIXME: this is repeated 3 times for a frame as it passes through
    blur3 = cv2.GaussianBlur(gray3, blur_size, 0)  # FIXME: this is repeated 3 times for a frame as it passes through

    # subtract the twoframes
    diff1 = cv2.absdiff(blur1, blur2)
    diff2 = cv2.absdiff(blur2, blur3)

    # combine
    combined = cv2.bitwise_and(diff1, diff2)

    # apply otsu thresholding
    ret, threshold = cv2.threshold(combined, 0, 255, cv2.THRESH_OTSU)
    # If Otsu's thresholding picks a low threshold due to low amount of foreground pixels
    if ret <= 8:
        _, threshold = cv2.threshold(combined, 24, 255, cv2.THRESH_BINARY)

    # apply morphological operations
    kernel = np.ones((3, 3), np.uint8)  # cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # processed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=8)
    dilated = cv2.dilate(threshold, kernel, iterations=9)
    processed = cv2.erode(dilated, kernel)

    return processed


def contour_center(contour) -> (int, int):
    """Return the center of the contour"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])
    return x, y


@dataclass(frozen=True)
class Rect:
    """Class for representing a rectangle in (x, y, width, height) form."""
    x: float
    y: float
    width: int
    height: int

    def center(self) -> (int, int):
        """
        :return: Center (x: int, y: int) of rectangle.
        """
        return int(self.x + self.width / 2), int(self.y + self.height)

    def center3D(self) -> (int, int, 1):
        """
        :return: Center (x: int, y: int, 1) of rectangle.
        """
        return *self.center(), 1

    def area(self) -> int:
        """
        :return: Area of rectangle.
        """
        return self.width * self.height


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

    @property
    def center_np(self):
        return np.array(self.center, dtype=np.float32)


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
    position: Tuple[float, float] = (size[0] / 2, size[1] / 2)  # in pixel coordinates

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
            prediction = self.kalman.predict().flatten().tolist()
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


class DoubleExponentialEstimator:
    """Provides an implementation for double exponential smoothing.
    """

    def __init__(self, initial_pos=Rect(0, 0, 0, 0), next_pos=Rect(0, 0, 0, 0)):
        """Create and initialize the estimator for forecasting.
        :param initial_pos: Initial observation position of Rectangle [top-left x, top-left y, width, height].
        :param next_pos: Next observation position of Rectangle [top-left x, top-left y, width, height].
        """
        self.__data_smoothing_factor = 0.9  # 0 <= data_smoothing_factor <= 1
        self.__trend_smoothing_factor = 0.25  # 0 <= trend_smoothing_factor <= 1

        self.__position_buffer = deque([initial_pos, next_pos], maxlen=2)

        self.__previous_smoothed = (self.__position_buffer[0].x, self.__position_buffer[0].y)
        self.__previous_trend = (self.__position_buffer[1].x - self.__position_buffer[0].x,
                                 self.__position_buffer[1].y - self.__position_buffer[0].y)

    def correct(self, position: Rect) -> None:
        """Add data to the position buffer.
        :param position: Bounding rectangle [top-left x, top-left y, width, height] of tracked object.
        """
        self.__position_buffer.append(position)

    def predict(self, t=1.0) -> Rect:
        """Forecasts a Rectangle [top-left x, top-left y, width, height] for time t=X.
        :param t: Time-step for which the forecast is made. Fractional time-steps are supported.
        :return: Predicted future bounding rectangle [top-left x, top-left y, width, height] of tracked object.
        """
        prev_pos = self.__position_buffer[-1]

        smoothed_previous_x, smoothed_previous_y = self.__previous_smoothed
        trend_previous_x, trend_previous_y = self.__previous_trend

        smoothed_x = self.__calculate_smoothed_value(prev_pos.x, smoothed_previous_x, trend_previous_x)
        smoothed_y = self.__calculate_smoothed_value(prev_pos.y, smoothed_previous_y, trend_previous_y)

        trend_x = self.__calculate_trend_estimate(smoothed_x, smoothed_previous_x, trend_previous_x)
        trend_y = self.__calculate_trend_estimate(smoothed_y, smoothed_previous_y, trend_previous_y)

        # The forecast follows an equation of the form: Prediction = b + tx,
        # Where t designates the time in the future for which the forecast is made.
        # i.e. t=1 means the forecast is for the next possible time-step.
        prediction_x = smoothed_x + t * trend_x
        prediction_y = smoothed_y + t * trend_y

        # Update the previous values of the estimates.
        self.__previous_smoothed = (smoothed_x, smoothed_y)
        self.__previous_trend = (trend_x, trend_y)

        # Return a Rect(top-left x, top-left y, width, height)
        return Rect(int(prediction_x) - 1, int(prediction_y) - 1, prev_pos.width + 2, prev_pos.height + 2)

    def __calculate_smoothed_value(self, observed_true: float, prev_smoothed: float, prev_trend: float) -> float:
        """Calculate the 'smoothed value' part of a double-exponential smoothing process.

        :param observed_true: Top-left corner of bounding rectangle which is assumed to be a true positive.
        :param prev_smoothed: Smoothed top-left corner of a bounding rectangle from previous time-step.
        :param prev_trend: Trend value from previous time-step.
        :return: Smoothed estimate of top-left corner of bounding rectangle.
        """
        return self.__data_smoothing_factor * observed_true + (1 - self.__data_smoothing_factor) * (
                prev_smoothed + prev_trend)

    def __calculate_trend_estimate(self, cur_smoothed: float, prev_smoothed: float, prev_trend: float) -> float:
        """Calculate the 'trend value' part of a double-exponential smoothing process.

        :param cur_smoothed: Smoothed value at current time-step.
        :param prev_smoothed: Smoothed value at previous time-step.
        :param prev_trend: Previous trend value.
        :return: New trend value.
        """
        return self.__trend_smoothing_factor * (cur_smoothed - prev_smoothed) + \
            (1 - self.__trend_smoothing_factor) * prev_trend


class Tracker:
    """
    Implements selection of the most probable ball contour from a list of contours.
    """

    def __init__(self):
        self.__candidate_history = deque(maxlen=7)  # deque(list[Rect], list[Rect], ...)

        # Dummy entries for initial start-up of the detector.
        dummy_candidate = Rect(0, 0, 0, 0)
        # Means that during frame 1, we had a single ball candidate: 'dummy candidate'
        self.__candidate_history.append([dummy_candidate])
        # Similarly, means that during frame 2, we also had single ball candidate: 'dummy candidate'
        self.__candidate_history.append([dummy_candidate])

        self.avg_area = 24 * 25  # Experimentally found nice constant
        self.__prev_best_dist = 0
        self.__dist_jump_cutoff = 100

        """ Mapping: goal: Rect -> (total_distance_required: float, from_rect: Rect, from_rect_layer_number: int) 
        total_distance_required gives the distance from layer 1 to reach current goal Rect. """
        self.__best_paths = dict()

    def select_most_probable_candidate(self, frame: np.ndarray, prediction: Rect) -> Rect:
        """
        Selects the contour from the frame that most likely appears to be a ball candidate.

        :param frame: Binarized video frame containing contours.
        :param prediction: Predicted contour of the ball in the frame.
        :returns: Contour in image corresponding to ball
        """

        # Clean the contours and store suitable candidates as ball candidates.
        self.__update_ball_candidates(frame, prediction)

        # Obtain the best ball candidate by searching for most continuous path
        # through the previous and up-to-current ball candidates.
        best_candidate = self.__find_shortest_path_candidate(prediction)

        # best_candidate is None in case of no candidates at all -> prediction is
        # selected as the most probable ball candidate.
        if best_candidate is None:
            best_candidate = prediction

        return best_candidate

    def __update_ball_candidates(self, frame: np.ndarray, prediction: Rect) -> None:
        """
        Process contours in the frame and update list of ball candidates from the contours.
        :param prediction: Predicted Rect ball candidate
        :param frame: Binarized video frame containing contours.
        """

        # Reduce noise by joining together nearby contours
        cleaned_contours = self.__join_contours(frame)

        # region DEBUG: Show detector view
        # frame_copy = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # for contour in cleaned_contours:
        #     utilities.draw_rect(frame_copy, contour, (255, 255, 0))
        # cv2.imshow("Tracker view", frame_copy)
        # endregion

        # Sort the contours in ascending order based on contour area
        # (Ideally the largest contour is the player and the smallest contour is the ball)
        cleaned_contours.sort(key=lambda rect: rect.area())
        # Filter tiny and excessively large contours
        ball_candidates = list(
            filter(lambda r: 0.3 * self.avg_area <= r.area() <= 3 * self.avg_area, cleaned_contours))

        # Throw away the biggest contour (most likely to be the player) only if such a big contour even exists
        # This prevents the undesirable action of discarding the real ball if it is the largest contour
        if ball_candidates:
            if cleaned_contours[-1].area() > self.avg_area * 1.5:
                ball_candidates = cleaned_contours[:(len(cleaned_contours) - 1)]

        self.__candidate_history.append(ball_candidates)

        # If all candidates were screened out, meaning there likely was no ball contour we automatically add the
        # prediction as a candidate at current time-step.
        # The above situation can arise due to occlusion (overlapping contours) or ball going out of frame.
        if not ball_candidates:
            self.__candidate_history[-1].extend([prediction])

    def __find_shortest_path_candidate(self, prediction) -> Rect:
        """
        Finds the shortest path through sequences of ball candidates.

        A most probable ball candidate is selected based on the idea that
        the movement of a squash ball follows a continuous path.

        :return: Ball candidate at the end of the shortest trajectory through the candidates.
        """
        self.__best_paths.clear()

        # Iterate over the collection of each frame's ball candidates
        # We start from 1 because we don't have anything to compare the first observation against
        for i in range(1, len(self.__candidate_history)):
            # for each candidate Rect in a candidate history "snapshot", assume that the most probable path goes
            # through the candidate
            for point_assumed_best in self.__candidate_history[i]:
                best_dist = sys.maxsize
                best_from_point = None  # Which point was the best to reach 'point_assumed_best'

                squareness = np.linalg.norm(point_assumed_best.width - point_assumed_best.height)
                # Calculate which was the most probable observation point we came from
                # i.e. this loops over the previous observation layer and compares distances between them.
                for from_point in self.__candidate_history[i - 1]:
                    dist = np.linalg.norm(
                        (point_assumed_best.x - from_point.x, point_assumed_best.y - from_point.y)) + squareness
                    # Remember the shortest distance and from which point it is achieved
                    if dist < best_dist:
                        best_dist = dist
                        best_from_point = from_point

                if best_from_point in self.__best_paths:
                    # Extend the path
                    self.__best_paths[point_assumed_best] = (self.__best_paths[best_from_point][0] + best_dist,
                                                             best_from_point,
                                                             self.__best_paths[best_from_point][2] + 1)
                else:
                    # Add a start entry for the path
                    self.__best_paths[point_assumed_best] = [best_dist, best_from_point, 1]

        best_dist = sys.maxsize
        best_point = None

        # Iterate over end-of-path rectangles
        for endpoint_rect in self.__best_paths:
            # Only end-Rects that are reached through traversing the full path are considered
            # If this check was not in place, then shorter noisy paths would corrupt the search.
            if self.__best_paths[endpoint_rect][2] == len(self.__candidate_history) - 1:
                dist = self.__best_paths[endpoint_rect][0]

                if best_dist > dist:
                    # self.__candidate_history[-1].extend([prediction])
                    best_dist = dist
                    best_point = endpoint_rect

        # Avoid sudden jumps in case the ball is lost for a frame or two
        if self.__prev_best_dist < best_dist - self.__dist_jump_cutoff:
            self.__prev_best_dist *= 1.2  # Inflate the distance as to not get stuck in a loop
            self.__candidate_history[-1].extend([prediction])
            return prediction

        self.__prev_best_dist = best_dist

        return best_point

    def __join_contours(self, frame: np.ndarray) -> list:
        """"
        :param frame: A preprocessed frame

        The method will receive a preprocessed image, which is likely to contain many contours due to the nature of
        the image segmentation process.
        Given the number of contours, it will be hard to identify which one is likely to be the ball and which are
        noise from the player movement.

        A heuristic to determine the ball contour is that the ball contour is mostly farther away from the player's body
        than the noisy segmentation of the player. Therefore, we will join all contours that are close to each other
        into one box.

        The result should ideally be one large and one small bounding box, that is, the player- and ball candidate
        respectively.
        """

        # Obtain all contours from the image
        contours, _ = cv2.findContours(cv2.Canny(frame, 0, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            bounding_boxes.append(cv2.boundingRect(contour))

        # Sort the bounding boxes according to their x-coordinate in increasing order
        bounding_boxes.sort(key=itemgetter(0))

        bounding_boxes = self.__join_nearby_bounding_boxes(bounding_boxes)

        return [Rect(*rect) for rect in bounding_boxes]

    def __join_nearby_bounding_boxes(self, bounding_boxes: List[list]) -> list:
        """
        :param bounding_boxes: Sorted list of bounding_boxes(rectangles)
        :return: List of rectangles [[x, y, width, height], ...]

        Many thanks to user HansHirse on StackOverflow.
        #https://stackoverflow.com/questions/55376338/how-to-join-nearby-bounding-boxes-in-opencv-python/55385454#55385454
        Algorithm has been adapted for squash-specific use.
        """

        join_distance_x = 5
        join_distance_y = 2 * join_distance_x
        processed = [False] * len(bounding_boxes)
        new_bounds = []

        for i, rect1 in enumerate(bounding_boxes):
            if not processed[i]:

                processed[i] = True
                current_x_min, current_y_min, current_x_max, current_y_max = self.__get_rectangle_contours(rect1)

                for j, rect2 in enumerate(bounding_boxes[(i + 1):], start=(i + 1)):

                    cand_x_min, cand_y_min, cand_x_max, cand_y_max = self.__get_rectangle_contours(rect2)

                    if current_x_max + join_distance_x >= cand_x_min:

                        if current_y_min < cand_y_min:
                            if not current_y_max + join_distance_y >= cand_y_min:
                                continue
                        else:
                            if not current_y_min - join_distance_y <= cand_y_max:
                                continue

                        processed[j] = True

                        # Reset coordinates of current rect
                        current_x_max = cand_x_max
                        current_y_min = min(current_y_min, cand_y_min)
                        current_y_max = max(current_y_max, cand_y_max)
                    else:
                        break
                new_bounds.append([current_x_min, current_y_min,
                                   current_x_max - current_x_min, current_y_max - current_y_min])

        return new_bounds

    @staticmethod
    def __get_rectangle_contours(rectangle: list) -> list:
        """
        Takes a rectangle and returns the coordinates of the top-left and
        bottom-right corner.

        :param rectangle: 4-tuple (top-left x, top-left y, width, height)
        :return: [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
        """
        x_min, y_min, width, height = rectangle
        x_max = x_min + width
        y_max = y_min + height

        return [x_min, y_min, x_max, y_max]


def draw_rect(frame: np.ndarray, rect: Rect, color: (int, int, int), line_width=2) -> None:
    cv2.rectangle(frame, (int(rect.x), int(rect.y)), (int(rect.x) + int(rect.width), int(rect.y) + int(rect.height)),
                  color, line_width)


# global state
i = 0
frame_c, frame_p, frame_pp = None, None, None
ball_candidates_prev = []
estimator = DoubleExponentialEstimator()
tracker = Tracker()
avg_ball_size = 24 * 25

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

    # get preliminary estimate
    prediction = estimator.predict(t=1)

    if prediction.x < 0 or prediction.y < 0:
        prediction = Rect(-prediction.width, -prediction.height, prediction.width, prediction.height)

    ball_bounding_box = tracker.select_most_probable_candidate(processed, prediction)
    estimator.correct(position=ball_bounding_box)

    # draw stuff
    draw_rect(frame, prediction, (0, 255, 0))
    draw_rect(frame, ball_bounding_box, (255, 0, 0))
    cv2.imshow("Frame", frame)
    cv2.imshow("After PreProcessing", processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
