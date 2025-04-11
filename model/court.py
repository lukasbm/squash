import os

import cv2
import numpy as np


def calibrate_camera(image):
    # corrects for my galaxy S24s 120° wide angle camera lens distortion
    # TODO: these have to be aquired for each camera (e.g. using a chessboard pattern)
    # use functions like cv2.findChessboardCorners() and cv2.calibrateCamera() and cv2.cornerSubPix()
    cameraMatrix = None  # 3x3 intrinsic camera matrix
    distCoeffs = None  # [k1, k2, p1, p2, k3] distortion coefficients
    return cv2.undistort(image, cameraMatrix, distCoeffs, None, cameraMatrix)


def get_court_lines(image):
    # blur
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert image from BGR to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # FIXME: way too strict!!!!

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

    # make the lines thinner
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

    # Use Canny edge detection to find edges in the mask.
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    # Apply Hough Line Transform to detect lines from the edges.
    # Here we use the probabilistic Hough Transform which directly returns endpoints.
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=50,
                            minLineLength=50,
                            maxLineGap=10)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    return lines, mask, edges

if __name__ == "__main__":
    # Step 1: Load the image
    image = cv2.imread(os.path.join("..", "videos", "frame.png"))  # Replace 'image.jpg' with your image file.
    if image is None:
        raise ValueError("Image not found. Check the file path.")

    # scale down
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Call the function to get court lines
    lines, mask_clean, edges = get_court_lines(image)
    output_image = image.copy()

    # Display the results
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing in green.
    else:
        print("No red lines were detected.")
    # cv2.imshow("Original Image", image)
    cv2.imshow("Red Mask", mask_clean)
    cv2.imshow("Edges", edges)
    cv2.imshow("Detected Lines", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
