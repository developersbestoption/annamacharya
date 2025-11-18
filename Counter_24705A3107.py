import cv2
import numpy as np
import subprocess

# -------------------------------
# Step 1: YouTube video URL
# -------------------------------
video_url = 'https://www.youtube.com/watch?v=aqz-KE-bpKQ'  # Big Buck Bunny trailer

# -------------------------------
# Step 2: Use yt-dlp to get direct video URL
# -------------------------------
try:
    # yt-dlp command to get best mp4 url
    result = subprocess.run(
        ['yt-dlp', '-f', 'best[ext=mp4]/best', '-g', video_url],
        capture_output=True, text=True, check=True
    )
    play_url = result.stdout.strip()
except Exception as e:
    print(f"Error fetching video URL: {e}")
    exit(1)

# -------------------------------
# Step 3: Open video stream
# -------------------------------
cap = cv2.VideoCapture(play_url)
if not cap.isOpened():
    print("Error: Cannot open video stream.")
    exit()

# -------------------------------
# Step 4: Process video frame by frame
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 1)

    cv2.imshow("YouTube Video Object Detection", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
