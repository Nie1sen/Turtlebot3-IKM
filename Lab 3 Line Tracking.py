import cv2
import numpy as np

def detect_and_track_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        [vx, vy, x0, y0] = cv2.fitLine(largest_contour, 
                                      cv2.DIST_L2, 0, 0.01, 0.01)
        
        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]
        
        center = (int(x0), int(y0))
        angle = np.arctan2(vy, vx) * 180 / np.pi
        
        return center, angle, (vx, vy)
    
    return None, None, None


# ---- Webcam Setup ----
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open video device")
    exit()

print("Webcam opened successfully. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    center, angle, direction = detect_and_track_line(frame)

    if center is not None:
        # Draw center
        cv2.circle(frame, center, 5, (0, 255, 0), -1)

        vx, vy = direction
        height, width = frame.shape[:2]
        length = max(width, height)

        x1 = int(center[0] - vx * length)
        y1 = int(center[1] - vy * length)
        x2 = int(center[0] + vx * length)
        y2 = int(center[1] + vy * length)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Optional: show angle text
        cv2.putText(frame,
                    f"Angle: {angle:.1f} deg",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2)

    cv2.imshow("Live Line Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()