import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty
import math
#part b
from sensor_msgs.msg import LaserScan
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# Load YOLO model (Ivan block model 1)
yolo_model = YOLO("best.pt")

image_pub = None

bridge = CvBridge()

target_visible = False
target_error = 0
obstacle_detected = False


def scan_callback(msg):
    global obstacle_detected
    # Define the safety distance in meters
    SAFETY_DIST = 0.25 
    
    # Check a 40-degree cone in front (20 deg left and 20 deg right)
    # TurtleBot3 index 0 does not exist. Indices 1 is left, 243 is right.
    front_ranges = msg.ranges[1:21] + msg.ranges[223:243]
    
    # Filter out 0.0 values (often noise/out of range) and check distances
    obstacle_detected = False

    scan_len = len(msg.ranges)

    # left side (1–20)
    for i in range(1,21):
        if i >= scan_len:
            break

        r = msg.ranges[i]
        if 0.0 < r < SAFETY_DIST:
            angle = msg.angle_min + i * msg.angle_increment
            rospy.logwarn(f"Obstacle detected at {math.degrees(angle):.1f} degrees")
            obstacle_detected = True
            break

    # right side (last 20 indices)
    if not obstacle_detected:
        for i in range(scan_len-20, scan_len):
            r = msg.ranges[i]
            if 0.0 < r < SAFETY_DIST:
                angle = msg.angle_min + i * msg.angle_increment
                rospy.logwarn(f"Obstacle detected at {math.degrees(angle):.1f} degrees")
                obstacle_detected = True
                break

def move():
    global image_pub
    image_pub = rospy.Publisher('/camera/image_processed', Image, queue_size=1)
    rospy.init_node('turtlebot3_autonomous_move', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(100)  # 100 Hz
    vel_msg = Twist()

    # speed step increments
    LIN_STEP = 0.01
    ANG_STEP = 0.1

    #lidar subscriber
    rospy.Subscriber('/scan', LaserScan, scan_callback)
    #camera subscriber
    rospy.Subscriber('/camera/image', Image, image_callback)

    while not rospy.is_shutdown():
        Kp = 0.002
        forward_speed = 0.05
        search_speed = 0.2

        if target_visible:
            vel_msg.linear.x = forward_speed
            vel_msg.angular.z = -Kp * target_error
        else:
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = search_speed

        if obstacle_detected and vel_msg.linear.x > 0:
            rospy.logwarn("SAFETY STOP: Obstacle in front!")
            vel_msg.linear.x = 0.0

        pub.publish(vel_msg)
        rate.sleep()

def image_callback(msg):
    global target_visible, target_error, image_pub

    # Convert ROS Image to OpenCV
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")

    # Run YOLO inference
    results = yolo_model(frame)

    # Start with no target visible
    target_visible = False

    # Iterate through detections
    for det in results[0].boxes:  # results[0].boxes contains all detections
        cls_id = int(det.cls[0])
        label = results[0].names[cls_id]

        # Get bounding box
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Draw rectangle + center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Compute target error
        width = frame.shape[1]
        center = width // 2
        target_error = cx - center
        target_visible = True

        # Only track the first phone detected
        break

    # Publish annotated frame to ROS topic
    image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))


if __name__ == '__main__':
    try:
        move()
    except rospy.ROSInterruptException:
        pass
