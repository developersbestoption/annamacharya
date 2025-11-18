import cv2
import numpy as np
import time

# ==========================================
# Load Haar Cascade Model
# ==========================================
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==========================================
# Load DNN Model (ResNet-10 SSD)
# ==========================================
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
dnn_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# ==========================================
# Start Webcam
# ==========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("Press 'q' to quit.")

# ==========================================
# Real-time Loop
# ==========================================
while True:
    start_time = time.time()  # For FPS calculation

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        break

    (h, w) = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------
    # Haar Cascade Detection
    # --------------------------------------
    haar_start = time.time()
    haar_result = frame.copy()
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    haar_time = (time.time() - haar_start) * 1000  # ms

    for (x, y, w_, h_) in faces:
        cv2.rectangle(haar_result, (x, y), (x + w_, y + h_), (0, 255, 0), 2)
        cv2.putText(haar_result, "Haar", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(haar_result, f"Haar Time: {haar_time:.1f} ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --------------------------------------
    # DNN Detection
    # --------------------------------------
    dnn_start = time.time()
    dnn_result = frame.copy()
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    dnn_time = (time.time() - dnn_start) * 1000  # ms

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(dnn_result, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"DNN {confidence*100:.1f}%"
            cv2.putText(dnn_result, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(dnn_result, f"DNN Time: {dnn_time:.1f} ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --------------------------------------
    # FPS Calculation
    # --------------------------------------
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    cv2.putText(haar_result, f"FPS: {fps:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(dnn_result, f"FPS: {fps:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --------------------------------------
    # Combine Results Side by Side
    # --------------------------------------
    combined = np.hstack((haar_result, dnn_result))
    cv2.imshow("Real-Time Face Detection - Haar (Left) vs DNN (Right)", combined)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# Cleanup
# ==========================================
cap.release()
cv2.destroyAllWindows()
