import cv2
import numpy as np
import subprocess
from google.colab.patches import cv2_imshow 


video_url = 'https://www.youtube.com/shorts/5GlgIRLQ96g?feature=share'  


try:
    
    result = subprocess.run(
        ['yt-dlp', '-f', 'best[ext=mp4]/best', '-g', video_url],
        capture_output=True, text=True, check=True
    )
    play_url = result.stdout.strip()
except Exception as e:
    print(f"Error fetching video URL: {e}")
    exit(1)


cap = cv2.VideoCapture(play_url)
if not cap.isOpened():
    print("Error: Cannot open video stream.")
    exit()


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

    cv2_imshow(frame) 

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
