#!/usr/bin/env python3

from std_msgs.msg import Int32
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
latest_lift_error = 0

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
    global latest_error, latest_lift_error, target_visible

    while not rospy.is_shutdown():

        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        results = yolo_model.predict(
            frame,
            device=0,
            #imgsz=320, # enable to reduce size if needed
            verbose=False
        )

        found = False
        best_error_x = 0
        best_error_y = 0

        if results and results[0].boxes is not None:

            highest_y = 999999

            for det in results[0].boxes:

                x1, y1, x2, y2 = map(int, det.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # choose HIGHEST block
                if cy < highest_y:
                    highest_y = cy

                    center_x = frame.shape[1] // 2
                    center_y = frame.shape[0] // 2

                    best_error_x = cx - center_x

                    # positive means block ABOVE center
                    best_error_y = center_y - cy

                    found = True

        with lock:
            latest_error = best_error_x
            latest_lift_error = best_error_y
            target_visible = found


# =========================
# MAIN CONTROL LOOP
# =========================
def control_loop(pub, lift_pub):
    rate = rospy.Rate(100)

    Kp = 0.002
    forward_speed = 0.07
    search_speed = 0.3
    find_speed = 0.15
    lift_deadband = 30 
    lift_speed = 30

    vel_msg = Twist()

    while not rospy.is_shutdown():

        with lock:
            error = latest_error
            lift_error = latest_lift_error
            visible = target_visible
            obstacle = obstacle_detected

        if obstacle:
            vel_msg.linear.x = 0
            vel_msg.angular.z = 0
            lift_pub.publish(Int32(data=0))

        elif visible:

            vel_msg.angular.z = -Kp * error
            vel_msg.angular.z = max(-find_speed,
                                    min(find_speed, vel_msg.angular.z))

            vel_msg.linear.x = forward_speed if abs(error) < 80 else 0.0

            lift_cmd = Int32()

            if lift_error > lift_deadband:
                lift_cmd.data = lift_speed

            elif lift_error < -(2*lift_deadband):
                lift_cmd.data = -lift_speed

            else:
                lift_cmd.data = 0

            lift_pub.publish(lift_cmd)

        else:
            vel_msg.linear.x = 0
            vel_msg.angular.z = search_speed
            lift_pub.publish(Int32(data=0))

        pub.publish(vel_msg)
        rate.sleep()


# =========================
# MAIN NODE
# =========================
def main():
    rospy.init_node("turtlebot3_yolo_tensorrt", anonymous=True)

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    lift_pub = rospy.Publisher("/lift_cmd", Int32, queue_size=10)

    #global image_pub    #debug option
    #image_pub = rospy.Publisher("/yolo/debug_image", Image, queue_size=1)    #debug option
    
    rospy.Subscriber(
        "/camera/image/compressed",
        CompressedImage,
        image_callback,
        queue_size=1,
        buff_size=2**24
    )

    # rospy.Subscriber("/scan", LaserScan, scan_callback)

    rospy.loginfo("YOLO TensorRT node started")

    # start YOLO thread
    t = threading.Thread(target=yolo_loop)
    t.daemon = True
    t.start()

    # control loop (main thread)
    control_loop(pub, lift_pub)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
