from ultralytics import YOLO
import cv2

# -------------------------------
# 1. Load YOLOv8 Model
# -------------------------------
# Load your custom-trained model
model_path = r"D:/TRAINING/ultralytics/runs/detect/train21/weights/best.pt"
model = YOLO(model_path)

# -------------------------------
# 2. Start Video Capture
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows stability

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("Press 'q' to quit...")

# -------------------------------
# 3. Live Detection Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Optional: Resize for faster inference
    # frame = cv2.resize(frame, (640, 480))

    # YOLO inference
    try:
        results = model(frame, conf=0.5)  # confidence threshold
    except Exception as e:
        print("❌ YOLO inference error:", e)
        break

    # Annotate frame
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("YOLOv8 Live Shape Detection", annotated_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 4. Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
