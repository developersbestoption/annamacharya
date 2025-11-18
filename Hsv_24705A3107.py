# Multi-Object CamShift Tracking with Trajectory Logging to CSV
# Auto-installs required packages if missing

import sys
import subprocess
import csv
import time

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Required packages
required_packages = ["opencv-python", "numpy"]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

import cv2
import numpy as np
import random
from collections import deque

# Capture video
cap = cv2.VideoCapture(0)

# Dictionary to hold tracking info for each object
tracked_objects = {}
object_id_counter = 0

# Termination criteria for CamShift
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Maximum trajectory length
MAX_TRAJECTORY_LENGTH = 50

# CSV file setup
csv_filename = "object_trajectories.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "ObjectID", "X", "Y"])

print(f"Trajectory logging started: {csv_filename}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (7, 7), 0)
    gray = blurred[:, :, 2]  # Value channel
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours of potential objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_windows = []

    for c in contours:
        if cv2.contourArea(c) < 500:  # Filter noise
            continue

        x, y, w, h = cv2.boundingRect(c)
        detected_windows.append((x, y, w, h))

        # Check if this object is already being tracked
        matched_id = None
        for obj_id, obj_info in tracked_objects.items():
            tx, ty, tw, th = obj_info['window']
            iou = (max(0, min(x + w, tx + tw) - max(x, tx)) *
                   max(0, min(y + h, ty + th) - max(y, ty))) / float(w*h + tw*th - max(0, min(x + w, tx + tw) - max(x, tx)) * max(0, min(y + h, ty + th) - max(y, ty)))
            if iou > 0.2:
                matched_id = obj_id
                break

        if matched_id is None:
            # New object detected
            object_id_counter += 1
            roi = hsv[y:y+h, x:x+w]
            roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            display_color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
            tracked_objects[object_id_counter] = {'window': (x, y, w, h), 'hist': roi_hist, 'color': display_color, 'trajectory': deque(maxlen=MAX_TRAJECTORY_LENGTH)}

    # Apply CamShift to each tracked object
    for obj_id, obj_info in tracked_objects.items():
        back_proj = cv2.calcBackProject([hsv], [0], obj_info['hist'], [0, 180], 1)
        ret_cam, obj_info['window'] = cv2.CamShift(back_proj, obj_info['window'], term_crit)
        pts = cv2.boxPoints(ret_cam)
        pts = np.int0(pts)
        frame = cv2.polylines(frame, [pts], True, obj_info['color'], 2)

        # Add current center to trajectory
        cx = int(ret_cam[0][0])
        cy = int(ret_cam[0][1])
        obj_info['trajectory'].append((cx, cy))

        # Draw trajectory
        for i in range(1, len(obj_info['trajectory'])):
            cv2.line(frame, obj_info['trajectory'][i-1], obj_info['trajectory'][i], obj_info['color'], 2)

        # Log trajectory to CSV
        timestamp = time.time()
        csv_writer.writerow([timestamp, obj_id, cx, cy])

    cv2.imshow("Multi-Object CamShift Tracking with CSV Logging", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"Trajectory logging finished. Data saved to {csv_filename}")
