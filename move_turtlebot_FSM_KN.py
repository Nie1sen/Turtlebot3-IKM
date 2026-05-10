#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, String
from ultralytics import YOLO
import numpy as np
import cv2
import threading
import time


# =========================
# GLOBAL STATE
# =========================
latest_frame = None
lock = threading.Lock()

model = YOLO("DropoffAndPackageModel.engine", task="detect")
model2 = YOLO("PackageTop.engine", task="detect")

# perception
block_seen = False
basket_seen = False
top_seen = False

block_error_x = 0
basket_error_x = 0
top_error_x = 0
block_area = 0

current_state = "SEARCH_BLOCK"


# =========================
# CONFIG
# =========================
CONF_THRESH = 0.5
LOST_TIMEOUT = 1.0

SEARCH_BLOCK = "SEARCH_BLOCK"
ALIGN_STACK = "ALIGN_STACK"
LIFT = "LIFT"
LOWER = "LOWER"
GRASP_TOP = "GRASP_TOP"
SEARCH_BASKET = "SEARCH_BASKET"
ALIGN_BASKET = "ALIGN_BASKET"
DROP = "DROP"
RETURN = "RETURN"


# =========================
# CAMERA
# =========================
def image_callback(msg):
    global latest_frame

    np_arr = np.frombuffer(msg.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    with lock:
        latest_frame = frame


# =========================
# YOLO THREAD
# =========================
def yolo_loop_main():

    global block_seen, basket_seen, block_error_x, block_area, basket_error_x

    last_seen_block = 0
    last_seen_basket = 0

    while not rospy.is_shutdown():

        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        results = model.predict(frame, device=0, verbose=False)

        h, w = frame.shape[:2]
        cx = w // 2

        block_seen = False
        basket_seen = False

        for det in results[0].boxes:

            conf = float(det.conf[0])
            if conf < CONF_THRESH:
                continue

            cls = int(det.cls[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            center_x = (x1 + x2) // 2
            area = (x2 - x1) * (y2 - y1)

            if cls == 1:  # block
                block_seen = True
                block_error_x = center_x - cx
                block_area = area
                last_seen_block = time.time()


            elif cls == 0:  # basket
                basket_seen = True
                basket_error_x = center_x - cx
                last_seen_basket = time.time()

    now = time.time()
    if now - last_seen_block > LOST_TIMEOUT:
        block_seen = False
    if now - last_seen_basket > LOST_TIMEOUT:
        basket_seen = False


def yolo_loop_top():

    global top_seen, top_error_x, current_state

    while not rospy.is_shutdown():

        with lock:
            state = current_state
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # ONLY RUN MODEL2 IN THESE STATES
        if state not in ["LIFT", "GRASP_TOP"]:
            top_seen = False
            time.sleep(0.05)
            continue

        results = model2.predict(frame, device=0, verbose=False)

        h, w = frame.shape[:2]
        cx = w // 2

        top_seen = False

        for det in results[0].boxes:

            conf = float(det.conf[0])
            if conf < CONF_THRESH:
                continue

            cls = int(det.cls[0])

            if cls == 0:  # top-of-block class
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                center_x = (x1 + x2) // 2

                top_seen = True
                top_error_x = center_x - cx
                break

# =========================
# CONTROL LOOP
# =========================
def control_loop(pub, lift_pub, claw_pub, state_pub, debug_pub):

    rate = rospy.Rate(50)

    state = SEARCH_BLOCK
    state_start = time.time()

    vel = Twist()
    lift = Int32()
    claw = Int32()

    Kp = 0.002
    SEARCH_SPEED = 0.3

    while not rospy.is_shutdown():

        debug_msg = ""

        # =========================
        # STATE MACHINE
        # =========================

        if state == SEARCH_BLOCK:

            vel.linear.x = 0
            vel.angular.z = SEARCH_SPEED

            debug_msg = "SEARCH_BLOCK | looking for block"

            if block_seen:
                state = ALIGN_STACK

        # -------------------------
        elif state == ALIGN_STACK:

            vel.angular.z = -Kp * block_error_x
            vel.angular.z = max(-0.3, min(0.3, vel.angular.z))

            vel.linear.x = 0.08 if block_area < 140000 else 0

            debug_msg = f"ALIGN_STACK | block_area={block_area}"

            if block_area > 140000:
                vel.linear.x = 0
                vel.angular.z = 0
                lift.data = 0
                state_start = time.time()
                state = LIFT

        # -------------------------
        elif state == LIFT:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 30
            claw.data = 0

            debug_msg = "LIFT | looking for top block"

            if time.time() - state_start > 5.0:
                state = LOWER
                state_start = time.time()

            if top_seen:
                state = GRASP_TOP
                state_start = time.time()

        # -------------------------
        elif state == LOWER:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = -30
            claw.data = 0

            debug_msg = "LOWER | looking for top block"

            if time.time() - state_start > 5.0:
                state = LIFT
                state_start = time.time()

            if top_seen:
                state = GRASP_TOP
                state_start = time.time()

        # -------------------------
        elif state == GRASP_TOP:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 0
            claw.data = -10

            debug_msg = "GRASP_TOP | Grabbing top block"
            
            if time.time() - state_start > 2.0:
                state = SEARCH_BASKET
                state_start = time.time()
            

        # -------------------------
        elif state == SEARCH_BASKET:

            vel.linear.x = 0
            vel.angular.z = SEARCH_SPEED

            debug_msg = "SEARCH_BASKET | looking for basket"

            if basket_seen:
                state = ALIGN_BASKET

        # -------------------------
        elif state == ALIGN_BASKET:

            vel.angular.z = -Kp * basket_error_x
            vel.angular.z = max(-0.3, min(0.3, vel.angular.z))
            vel.linear.x = 0.08

            debug_msg = f"ALIGN_BASKET | error={basket_error_x}"

            if abs(basket_error_x) < 20:
                state_start = time.time()
                state = DROP

        # -------------------------
        elif state == DROP:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 0
            claw.data = 10

            debug_msg = "DROP | releasing object"

            if time.time() - state_start > 2.0:
                state = RETURN
                state_start = time.time()

        # -------------------------
        elif state == RETURN:

            vel.linear.x = -0.05
            vel.angular.z = 0
            lift.data = -10
            claw.data = 0

            debug_msg = "RETURN | backing up"

            if time.time() - state_start > 5.0:
                state = SEARCH_BLOCK



        # =========================
        # PUBLISH
        # =========================
        with lock:
            global current_state
            current_state = state 

        pub.publish(vel)
        lift_pub.publish(lift)
        claw_pub.publish(claw)

        state_pub.publish(String(data=state))
        debug_pub.publish(String(data=debug_msg))

        rate.sleep()


# =========================
# MAIN
# =========================
def main():

    rospy.init_node("fsm_yolo_robot")

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    lift_pub = rospy.Publisher("/lift_cmd", Int32, queue_size=10)
    claw_pub = rospy.Publisher("/claw_cmd", Int32, queue_size=10)
    state_pub = rospy.Publisher("/fsm_state", String, queue_size=10)
    debug_pub = rospy.Publisher("/fsm_debug", String, queue_size=10)

    rospy.Subscriber("/camera/image/compressed",
                     CompressedImage,
                     image_callback,
                     queue_size=1,
                     buff_size=2**24)

    t1 = threading.Thread(target=yolo_loop_main)
    t1.daemon = True
    t1.start()

    t2 = threading.Thread(target=yolo_loop_top)
    t2.daemon = True
    t2.start()

    control_loop(pub, lift_pub, claw_pub, state_pub, debug_pub)


if __name__ == "__main__":
    main()