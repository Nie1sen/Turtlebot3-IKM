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

model = YOLO("TrashAndBoxes.engine", task="detect")
model2 = YOLO("topBlockModel.engine", task="detect")

# perception
block_seen = False
basket_seen = False
top_seen = False

block_error_x = 0
basket_error_x = 0
top_error_x = 0

block_area = 0
basket_area = 0

BASE_SPEED = 0.10
K_AREA = 8e-7     # tune this
MIN_SPEED = 0.02

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
LIFT_TOP = "LIFT_TOP"
SEARCH_BASKET = "SEARCH_BASKET"
ALIGN_BASKET = "ALIGN_BASKET"
LIFT_BASKET = "LIFT_BASKET"
MOVE_TO_BASKET = "MOVE_TO_BASKET"
DROP = "DROP"
MOVE_FROM_BASKET = "MOVE_FROM_BASKET"
RETURN = "RETURN"


# =========================
# CAMERA CALLBACK
# =========================
def image_callback(msg):
    global latest_frame

    np_arr = np.frombuffer(msg.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    with lock:
        latest_frame = frame


# =========================
# YOLO LOOP MAIN MODEL
# =========================
def yolo_loop_main():
    global block_seen, basket_seen
    global block_error_x, basket_error_x
    global block_area, basket_area
    global current_state

    last_seen_block = 0
    last_seen_basket = 0

    allowed_states = [SEARCH_BLOCK, ALIGN_STACK, SEARCH_BASKET, ALIGN_BASKET]

    while not rospy.is_shutdown():

        with lock:
            state = current_state
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # ONLY RUN MODEL1 IN THESE STATES
        if state not in allowed_states:
            with lock:
                block_seen = False
                basket_seen = False
                block_area = 0
                basket_area = 0
            time.sleep(0.05)
            continue

        results = model.predict(frame, device=0, verbose=False)

        h, w = frame.shape[:2]
        cx_frame = w // 2

        local_block_seen = False
        local_basket_seen = False
        local_block_area = 0
        local_basket_area = 0
        local_block_error = 0
        local_basket_error = 0

        if results and results[0].boxes is not None:
            for det in results[0].boxes:

                conf = float(det.conf[0])
                if conf < CONF_THRESH:
                    continue

                cls = int(det.cls[0])
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cx = (x1 + x2) // 2
                area = (x2 - x1) * (y2 - y1)

                if cls == 0:  # basket
                    local_basket_seen = True
                    local_basket_error = cx - cx_frame
                    local_basket_area = max(local_basket_area, area)
                    last_seen_basket = time.time()

                elif cls == 1:  # block
                    local_block_seen = True
                    local_block_error = cx - cx_frame
                    local_block_area = max(local_block_area, area)
                    last_seen_block = time.time()

        now = time.time()

        if now - last_seen_block > LOST_TIMEOUT:
            local_block_seen = False
            local_block_area = 0

        if now - last_seen_basket > LOST_TIMEOUT:
            local_basket_seen = False
            local_basket_area = 0

        # write back safely
        with lock:
            block_seen = local_block_seen
            basket_seen = local_basket_seen
            block_area = local_block_area
            basket_area = local_basket_area
            block_error_x = local_block_error
            basket_error_x = local_basket_error


# =========================
# YOLO LOOP TOP MODEL
# =========================
def yolo_loop_top():
    global top_seen, top_error_x, current_state

    last_seen_top = 0

    while not rospy.is_shutdown():

        with lock:
            state = current_state
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # ONLY RUN MODEL2 IN THESE STATES
        if state not in [LIFT, LOWER, GRASP_TOP]:
            top_seen = False
            time.sleep(0.05)
            continue

        results = model2.predict(frame, device=0, verbose=False)

        h, w = frame.shape[:2]
        cx_frame = w // 2

        top_seen = False

        if results and results[0].boxes is not None:
            for det in results[0].boxes:

                conf = float(det.conf[0])
                if conf < CONF_THRESH:
                    continue

                cls = int(det.cls[0])

                # model2 only has top block as class 0
                if cls == 0:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    cx = (x1 + x2) // 2

                    top_seen = True
                    top_error_x = cx - cx_frame
                    last_seen_top = time.time()
                    break

        # LOST TIMEOUT FILTERING
        now = time.time()
        if now - last_seen_top > LOST_TIMEOUT:
            top_seen = False


# =========================
# FSM CONTROL LOOP
# =========================
def control_loop(pub, lift_pub, claw_pub, state_pub, debug_pub):

    global current_state
    global block_seen, basket_seen, top_seen
    global block_error_x, basket_error_x
    global block_area, basket_area

    rate = rospy.Rate(30)

    state = SEARCH_BLOCK
    last_state = None
    state_start = time.time()

    vel = Twist()
    lift = Int32()
    claw = Int32()

    Kp = 0.002
    SEARCH_SPEED = 0.3

    total_lift_time = 0.0

    while not rospy.is_shutdown():

        debug_msg = (
            f"state={state} | "
            f"block_seen={block_seen}, top_seen={top_seen}, basket_seen={basket_seen} | "
            f"block_area={block_area}, basket_area={basket_area} | "
            f"block_err={block_error_x}, basket_err={basket_error_x} | "
            f"lift_time={total_lift_time:.2f}"
        )

        # =========================
        # STATE MACHINE
        # =========================

        if state == SEARCH_BLOCK:

            vel.linear.x = 0
            vel.angular.z = SEARCH_SPEED
            lift.data = 0
            claw.data = 0

            if block_seen:
                state = ALIGN_STACK

        elif state == ALIGN_STACK:

            vel.angular.z = -Kp * block_error_x
            vel.angular.z = max(-0.3, min(0.3, vel.angular.z))

            # only drive forward if aligned enough
            if abs(block_error_x) < 10:

                if block_area < 140000:
                    vel.linear.x = BASE_SPEED - K_AREA * block_area
                    vel.linear.x = max(MIN_SPEED, min(BASE_SPEED, vel.linear.x))
                else:
                    vel.linear.x = 0.0
                    vel.angular.z = 0.0
                    state_start = time.time()
                    state = LIFT

            else:
                # not aligned yet: just turn in place
                vel.linear.x = 0.0

        elif state == LIFT:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 50
            claw.data = 0

            if time.time() - state_start > 45.0:
                total_lift_time += time.time() - state_start
                state_start = time.time()
                state = LOWER

            if top_seen:
                total_lift_time += time.time() - state_start
                state_start = time.time()
                state = GRASP_TOP

        elif state == LOWER:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = -50
            claw.data = 0

            if time.time() - state_start > 45.0:
                total_lift_time -= time.time() - state_start
                state_start = time.time()
                state = LIFT

            if top_seen:
                total_lift_time -= time.time() - state_start
                state_start = time.time()
                state = GRASP_TOP

        elif state == GRASP_TOP:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 0
            claw.data = 20

            if time.time() - state_start > 3.5:
                state = LIFT_TOP
                state_start = time.time()

        elif state == LIFT_TOP:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 50
            claw.data = 0

            if time.time() - state_start > 1.0:
                state = SEARCH_BASKET
                total_lift_time += time.time() - state_start
                state_start = time.time()

        elif state == SEARCH_BASKET:

            vel.linear.x = 0
            vel.angular.z = SEARCH_SPEED
            lift.data = 0
            claw.data = 0

            if basket_seen:
                state = ALIGN_BASKET

        elif state == ALIGN_BASKET:
            vel.angular.z = -Kp * basket_error_x
            vel.angular.z = max(-0.3, min(0.3, vel.angular.z))

            # only drive forward if aligned enough
            if abs(basket_error_x) < 10:

                if basket_area < 100000:
                    vel.linear.x = BASE_SPEED - K_AREA * basket_area
                    vel.linear.x = max(MIN_SPEED, min(BASE_SPEED, vel.linear.x))
                else:
                    vel.linear.x = 0.0
                    vel.angular.z = 0.0
                    state_start = time.time()
                    state = LIFT_BASKET
        
            else:
                # not aligned yet: just turn in place
                vel.linear.x = 0.0

        elif state == LIFT_BASKET:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 50
            claw.data = 0

            # raise lift to match the previously recorded lift time
            if total_lift_time > 45.0:
                state = MOVE_TO_BASKET
                state_start = time.time()
            else:
                total_lift_time += time.time() - state_start
                state_start = time.time()

        elif state == MOVE_TO_BASKET:

            vel.linear.x = 0.08
            vel.angular.z = 0
            lift.data = 0
            claw.data = 0

            if time.time() - state_start > 3.0:
                state = DROP
                state_start = time.time()

        elif state == DROP:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 0
            claw.data = -20

            if time.time() - state_start > 3.5:
                claw.data = 0
                state = MOVE_FROM_BASKET
                state_start = time.time()

        elif state == MOVE_FROM_BASKET:

            vel.linear.x = -0.08
            vel.angular.z = 0
            lift.data = 0
            claw.data = 0

            if time.time() - state_start > 3.0:
                state = RETURN
                state_start = time.time()

        elif state == RETURN:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = -50
            claw.data = 0

            if time.time() - state_start > 45.0:
                lift.data = 0

                total_lift_time = 0.0
                state = SEARCH_BLOCK
                state_start = time.time()

        # =========================
        # UPDATE CURRENT STATE (for YOLO2 gating)
        # =========================
        with lock:
            current_state = state

        # =========================
        # PUBLISH
        # =========================
        pub.publish(vel)
        lift_pub.publish(lift)
        claw_pub.publish(claw)

        if state != last_state:
            rospy.loginfo(f"STATE CHANGE: {last_state} -> {state}")
            state_pub.publish(String(data=state))
            last_state = state

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

    rospy.Subscriber(
        "/camera/image/compressed",
        CompressedImage,
        image_callback,
        queue_size=1,
        buff_size=2**24
    )

    t1 = threading.Thread(target=yolo_loop_main)
    t1.daemon = True
    t1.start()

    t2 = threading.Thread(target=yolo_loop_top)
    t2.daemon = True
    t2.start()

    control_loop(pub, lift_pub, claw_pub, state_pub, debug_pub)


if __name__ == "__main__":
    main()
