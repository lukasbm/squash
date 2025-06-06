import os

import cv2
import numpy as np

# https://www.worldsquash.org/wp-content/uploads/2021/08/171128_Court-Specifications.pdf
# Dimensions of an International Doubles Squash Court
class Court:
    # (horizontal, vertical, height) toward back wall is positive. right is positive.
    REFERENCE_POINTS = {
        # marking line corners
        "T": (0, 0, 0),  # half court line meets short line
        "short line left wall": (-3810, 0, 0),  # short line meets left wall
        "short line right wall": (3810, 0, 0),  # short line meets right wall
        "short line left box": (-2185, 0, 0),  # left box meets short line
        "short line right box": (2185, 0, 0),  # right box meets short line
        "center door": (0, -4285, 0),  # start of half court line
        "left box corner": (-2185, -1650, 0),  # inside corner left box
        "right box corner": (2185, -1650, 0),  # inside corner right box
        "left box wall": (-3810, -1650, 0),  # left box wall
        "right box wall": (3810, -1650, 0),  # right box wall
        # just the corners of the court
        "front left": (-3810, 5465, 0),  # front left corner
        "front right": (3810, 5465, 0),  # front right corner
        "back left": (-3810, -4285, 0),  # back left corner
        "back right": (3810, -4285, 0),  # back right corner
        # front wall (back) marking line points
        "tin left": (-3810, 5465, 455),  # left wall meets tin
        "tin right": (3810, 5465, 455),  # right wall meets tin
        "service left": (-3810, 5465, 1805),  # left wall meets service line
        "service right": (3810, 5465, 1805),  # right wall meets service line
        "front wall left": (-3810, 5465, 4595),  # left wall meets front wall line
        "front wall right": (3810, 5465, 4595),  # right wall meets front wall line
        # back wall (door) marking line points
        "back wall left": (-3810, -4285, 2155),  # left wall meets back wall line
        "back wall right": (3810, -4285, 2155),  # right wall meets back wall line
        # left wall marking line points
        "left wall top": (-3810, 5465, 4595),  # left wall meets front wall line
        "left wall bottom": (-3810, -4285, 2155),  # left wall meets back wall line
        # right wall marking line points
        "right wall top": (3810, 5465, 4595),  # right wall meets front wall line
        "right wall bottom": (3810, -4285, 2155),  # right wall meets back wall line
    }

    def __init__(self, corners_pixel_space):
        # FIXME: need to match corners_pixel_space to proper key!!!
        H, status = cv2.findHomography(corners_pixel_space, self.REFERENCE_POINTS.values())
        self.H = H

    def pixel_to_world(self, u, v):
        # convert pixel coordinates to world coordinates
        # NOTE: only works for objects on the floor, e.g. shoes
        pixel_point = np.array([[u], [v], [1]])
        world_point_homogeneous = np.dot(self.H, pixel_point)
        # normalize to convert from homogenous to cartesian coordinates
        world_point = world_point_homogeneous / world_point_homogeneous[2]
        return world_point[0:2].flatten()  # return (x, y) coordinates


def calibrate_camera(image):
    # corrects for my galaxy S24s 120° wide angle camera lens distortion
    # TODO: these have to be aquired for each camera (e.g. using a chessboard pattern)
    # use functions like cv2.findChessboardCorners() and cv2.calibrateCamera() and cv2.cornerSubPix()
    cameraMatrix = None  # 3x3 intrinsic camera matrix
    distCoeffs = None  # [k1, k2, p1, p2, k3] distortion coefficients
    return cv2.undistort(image, cameraMatrix, distCoeffs, None, cameraMatrix)


def get_court_mask(image):
    # blur
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert image from BGR to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Step 2: Define HSV ranges for red.
    # Note: Red hue is at the low end (0-10 degrees) and the high end (160-180 degrees).
    lower_red1 = np.array([0, 85, 50])
    upper_red1 = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 85, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the masks to capture the full red range
    mask = cv2.bitwise_or(mask1, mask2)

    return mask


def get_court_lines(image):
    mask = get_court_mask(image)

    # grow the lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Apply Hough Line Transform to detect lines from the edges.
    # Here we use the probabilistic Hough Transform which directly returns endpoints.
    lines = cv2.HoughLines(mask_clean,
                           rho=1,
                           theta=np.pi / 90,
                           threshold=125)

    return lines, mask_clean


def get_court_corners(image):
    mask = get_court_mask(image)

    # corner detector
    # corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.01, minDistance=10)

    # corner detector (harris)
    cv2.corner
    corners = cv2.cornerHarris(mask, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)
    corners = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)[1]

    # non maximal suppression
    corners = cv2.dilate(corners, None)
    corners = np.int0(corners)

    return corners


if __name__ == "__main__":
    # Step 1: Load the image
    image = cv2.imread(os.path.join("..", "videos", "frame.png"))  # Replace 'image.jpg' with your image file.
    if image is None:
        raise ValueError("Image not found. Check the file path.")

    # just testing with manual determination of points (only frame.png)
    pixel_space_points_image = {
        # marking line corners
        "T": (416, 350),  # half court line meets short line
        "short line left wall": (90, 358),  # short line meets left wall
        "short line right wall": (720, 340),  # short line meets right wall
        "short line left box": (257, 354),  # left box meets short line
        "short line right box": (570, 344),  # right box meets short line
        "left box corner": (170, 476),  # inside corner left box
        "right box corner": (639, 454),  # inside corner right box
        "right box wall": (860, 445),  # right box wall
        # just the corners of the court
        "front left": (270, 231),  # front left corner
        "front right": (569, 228),  # front right corner
        # front wall (back) marking line points
        "tin left": (269, 212),  # left wall meets tin
        "tin right": (569, 209),  # right wall meets tin
        "service left": (262, 145),  # left wall meets service line
        "service right": (573, 144),  # right wall meets service line
    }

    # scale down
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    # Call the function to get court lines
    lines, mask_clean = get_court_lines(image)
    output_image = image.copy()
    # Display the results
    print(f"Detected {len(lines)} lines.")
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing in green.

    # get court corners
    # corners =

    # cv2.imshow("Original Image", image)
    cv2.imshow("Red Mask", mask_clean)
    cv2.imshow("Detected Lines", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
