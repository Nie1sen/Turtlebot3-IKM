import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open video device")
    exit()

print("Webcam opened successfully. Press 'q' to exit.")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Turtlebot Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# OpenCV is ideal for simple, resource-constrained tasks,
# while YOLO excels in complex, high-accuracy, multi-object scenarios,
# but is farm more resource intensive. The advantages and disadvantages
# depend on the specific requirements of the application, such as, 
# accuracy needs, computational resources, and real-time performance constraints.
