#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import cv2
import math
import threading
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
import numpy as np


# =========================
# GLOBAL STATE (shared)
# =========================
latest_frame = None
latest_error = 0
target_visible = False
obstacle_detected = False

#image_pub = None    #debug option

lock = threading.Lock()


# =========================
# YOLO MODEL (TensorRT)
# =========================
yolo_model = YOLO("best.engine", task="detect")
bridge = CvBridge()


# =========================
# LIDAR CALLBACK
# =========================
def scan_callback(msg):
    global obstacle_detected

    SAFETY_DIST = 0.25
    scan_len = len(msg.ranges)

    detected = False

    # front-left
    for i in range(1, 21):
        if i >= scan_len:
            break
        r = msg.ranges[i]
        if 0.0 < r < SAFETY_DIST:
            detected = True
            break

    # front-right
    if not detected:
        for i in range(scan_len - 20, scan_len):
            r = msg.ranges[i]
            if 0.0 < r < SAFETY_DIST:
                detected = True
                break

    with lock:
        obstacle_detected = detected


# =========================
# CAMERA + YOLO CALLBACK
# =========================
def image_callback(msg):
    global latest_frame

    np_arr = np.frombuffer(msg.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    with lock:
        latest_frame = frame



def yolo_loop():
    global latest_error, target_visible
    #global image_pub    #debug option

    while not rospy.is_shutdown():

        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        results = yolo_model.predict(
            frame,
            device=0,
            imgsz=320,     # BIG SPEED BOOST
            verbose=False
        )

        found = False
        error = 0

        if results and results[0].boxes is not None:
            for det in results[0].boxes:

                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cx = (x1 + x2) // 2

                center = frame.shape[1] // 2
                error = cx - center

                found = True
                
                # publish debug image
                # image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))    #debug option
                break

        with lock:
            latest_error = error
            target_visible = found


# =========================
# MAIN CONTROL LOOP
# =========================
def control_loop(pub):
    rate = rospy.Rate(100)

    Kp = 0.002
    forward_speed = 0.05
    search_speed = 0.1

    vel_msg = Twist()

    while not rospy.is_shutdown():

        with lock:
            error = latest_error
            visible = target_visible
            obstacle = obstacle_detected

        if obstacle:
            vel_msg.linear.x = 0
            vel_msg.angular.z = 0

        elif visible:
            vel_msg.angular.z = -Kp * error
            vel_msg.linear.x = forward_speed if abs(error) < 80 else 0.0

        else:
            vel_msg.linear.x = 0
            vel_msg.angular.z = search_speed

        pub.publish(vel_msg)
        rate.sleep()


# =========================
# MAIN NODE
# =========================
def main():
    rospy.init_node("turtlebot3_yolo_tensorrt", anonymous=True)

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    
    #global image_pub    #debug option
    #image_pub = rospy.Publisher("/yolo/debug_image", Image, queue_size=1)    #debug option
    
    rospy.Subscriber(
        "/camera/image/compressed",
        CompressedImage,
        image_callback,
        queue_size=1,
        buff_size=2**24
    )

    rospy.Subscriber("/scan", LaserScan, scan_callback)

    rospy.loginfo("YOLO TensorRT node started")

    # start YOLO thread
    t = threading.Thread(target=yolo_loop)
    t.daemon = True
    t.start()

    # control loop (main thread)
    control_loop(pub)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
