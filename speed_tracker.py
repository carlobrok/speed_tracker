import numpy as np
from random import randint
import sys
import os
import cv2
import csv
import argparse

# parse args
parser = argparse.ArgumentParser(description="Script to track objects and estimate their speed in videos that move in a straight path perpendicular to the camera.",
                                 formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=60, width=150))

parser.add_argument('video_in', help="video input file path")
parser.add_argument(
    'csv_out', help="csv output file basename (e.g. 'speed_export')")
parser.add_argument(
    'f', type=float, help="length of a single frame in seconds")
parser.add_argument('-r', type=int, help="rotation of video", default=0)
parser.add_argument(
    '-l', type=float, help="length of reference line in meter", default=2.0)

args, leftovers = parser.parse_known_args()
videoPath = args.video_in
frame_time = args.f
rotation = args.r


def check_esc_exit(key):
    if key == 27:
        sys.exit(1)


def check_r_reset(key):
    if key == ord("r"):
        reset_points()


def show_image():
    cv2.imshow('speed tracker', frame)


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


refPts = []
scale = -1
ref_length = args.l


def select_ref_line(event, x, y, flags, param):
    # grab references to the global variables
    global refPts, frame, frame_copy, scale
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPts = [(x, y)]
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPts.append((x, y))
        # draw a rectangle around the region of interest
        scale = distance(refPts[0], refPts[1]) / 2
        print("Selected " + str(ref_length) +
              "m reference line. Scale: {} px/m".format(scale))

    if len(refPts) == 1:
        frame = frame_copy.copy()
        cv2.line(frame, refPts[0], (x, y), (0, 255, 0), 4)
        show_image()


data_saved = False


def save_speeds():
    global data_saved

    csv_name = args.csv_out + '%s.csv'

    file_rotation_number = 0
    while os.path.exists(csv_name % file_rotation_number):
        file_rotation_number += 1

    csv_name = csv_name % file_rotation_number

    print('csv: ' + csv_name)
    with open(csv_name, 'w') as csvfile:
        fieldnames = ['time', 'speed', 'distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()

        for i in range(len(center_points) - 1):
            dist = distance(center_points[i], center_points[i + 1]) / scale
            speed = dist / frame_time
            writer.writerow({'speed': str(speed).replace('.', ','),
                             'time': str(i * 0.02).replace('.', ','),
                             'distance': str(dist).replace('.', ',')})

        print('csv saved.')
        data_saved = True


def select_bbox():
    return cv2.selectROI('speed tracker', frame)


def reset_points():
    global center_points
    center_points = []


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

#cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)
cv2.namedWindow('speed tracker', cv2.WINDOW_NORMAL)
cv2.resizeWindow('speed tracker', 850, 850)

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
    print('Failed to read video: ' + videoPath)
    sys.exit(1)

frame = rotate_bound(frame, rotation)

print(">  Press r to rotate video clockwise, press space to continue.")
while True:

    show_image()
    k = cv2.waitKey(1) & 0xFF
    if k == 114 or k == 82:
        frame = rotate_bound(frame, 90)
        rotation = (rotation + 90) % 360
    if (k == 32):  # space is pressed
        break
    check_esc_exit(k)

# select scale of video
frame_copy = frame.copy()
cv2.setMouseCallback('speed tracker', select_ref_line)

print(">  Click and drag to select ~ " +
      str(ref_length) + "m reference line, press space to continue.")
while True:

    k = cv2.waitKey(1) & 0xFF
    if (k == 32):  # space is pressed
        break
    check_esc_exit(k)

if scale < 0:
    print("Error: no scale selected")
    sys.exit(1)

frame = frame_copy.copy()

# draw bounding boxes over objects
# selectROI's default behaviour is to draw box starting from the center
# when fromCenter is set to false, you can draw box starting from top left corner
print(">  Select ROI of object")
bbox = select_bbox()
if bbox == (0, 0, 0, 0):
    print("Error: No ROI selected!")
    sys.exit(1)

    sys.exit(1)
color = (randint(64, 255), randint(64, 255), randint(64, 255))
tracker = cv2.TrackerCSRT_create()
success = tracker.init(frame, bbox)

print(">  Press space to pause, then s to save data to csv file and reset cached points")
print(">  Press r to reset cached points")

center_points = []

# Process video and track objects
while cap.isOpened():
    # get frame and rotate
    success, frame = cap.read()
    if not success:
        break
    if rotation != 0:
        frame = rotate_bound(frame, rotation)

    ok, bbox = tracker.update(frame)

    # if box found draw box, calculate center, save center point
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        center_points.append(
            (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # draw all center points
    for center in center_points:
        cv2.circle(frame, center, 3, (0, 0, 255), -1)

    # show frame
    show_image()

    k = cv2.waitKey(1) & 0xFF
    #    quit on ESC button
    check_esc_exit(k)
    # r -> reset points
    check_r_reset(k)

    # pause with space
    if k == 32:
        print("Press s to save data / space to continue ")
        k = cv2.waitKey(1) & 0xFF
        while True:
            k = cv2.waitKey(0) & 0xFF
            # space -> continue
            if (k == 32):
                break
            # s -> save data
            elif (k == ord("s")):
                save_speeds()
                reset_points()
            # r -> reset points
            elif k == ord("r"):
                reset_points()
            # Exit program
            check_esc_exit(k)

if not data_saved:
    print("Press s to save cached data")
    if cv2.waitKey(0) & 0xFF == ord("s"):
        save_speeds()

cap.release()
cv2.destroyAllWindows()
