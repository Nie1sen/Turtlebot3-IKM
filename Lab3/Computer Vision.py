import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model_num = 1  # Change this number to select different YOLO models (1-5)
match model_num:    
    case 1:       model = YOLO("yolov8n.pt")   # nano model (fastest)
    case 2:       model = YOLO("yolov8s.pt")   # small model (more accurate)
    case 3:       model = YOLO("yolov8m.pt")   # medium model (balance of speed and accuracy)
    case 4:       model = YOLO("yolov8l.pt")   # large model (more accurate but slower)
    case 5:       model = YOLO("yolov8x.pt")   # extra large model (most accurate but slowest)

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open video device")
    exit()

print("Webcam opened successfully. Press 'q' to exit.")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Draw bounding boxes + labels
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("YOLO Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
