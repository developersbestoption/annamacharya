import cv2
import yt_dlp
import numpy as np

def detect_objects_from_youtube(youtube_url):
    """
    Performs object detection using contour detection and bounding boxes on a YouTube video.

    Args:
        youtube_url (str): The URL of the YouTube video.
    """
    try:
        # Get the YouTube video stream URL using yt-dlp
        yt_opts = {'format': 'bestvideo[ext=mp4]'}
        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            stream_url = info_dict['url']
    except yt_dlp.utils.DownloadError as e:
        print(f"Error: Could not retrieve video information. {e}")
        return

    # Create a VideoCapture object from the stream URL
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Could not open video stream from {youtube_url}")
        return

    # Create a background subtractor
    # This helps in isolating moving objects from a static background.
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    print("Processing video... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Apply background subtraction to get a foreground mask
        fg_mask = back_sub.apply(frame)

        # Apply a binary threshold to the foreground mask to get a cleaner image
        # This simplifies the image to pure black (background) and white (foreground).
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours of the detected objects
        # `cv2.findContours` finds all the boundaries of objects in the binary image.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour to find valid objects
        for contour in contours:
            # Filter out small contours that are likely noise
            if cv2.contourArea(contour) > 500:
                # Get the bounding box for the contour
                (x, y, w, h) = cv2.boundingRect(contour)
                
                # Draw the bounding box on the original frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame with bounding boxes
        cv2.imshow('Object Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage: Replace with your desired YouTube video link
if __name__ == '__main__':
    youtube_link = 'https://www.youtube.com/watch?v=Or-XHvRZFq0&pp=ygUadG9tIGFuZCBqZXJyeSBzaG9ydHMgZnVubnk%3D' # Example link
    detect_objects_from_youtube(youtube_link)
