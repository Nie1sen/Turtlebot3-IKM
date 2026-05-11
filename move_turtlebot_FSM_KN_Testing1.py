#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, String
import time
import sys
import termios
import tty
import threading

# =========================
# CONFIG
# =========================
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
# SIMULATED PERCEPTION
# =========================
block_seen = False
basket_seen = False
top_seen = False

block_error_x = 0
basket_error_x = 0
block_area = 0
basket_area = 0
total_lift_time = 0.0
BASE_SPEED = 0.10
K_AREA = 8e-7     # tune this
MIN_SPEED = 0.02


# =========================
# KEYBOARD INPUT
# =========================
def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def print_controls():
    print("\n=== FSM DEBUG CONTROLS ===")
    print("b = toggle block_seen")
    print("t = toggle top_seen")
    print("k = toggle basket_seen")
    print("+ = increase block_area")
    print("- = decrease block_area")
    print("a/d = block error left/right")
    print("j/l = basket error left/right")
    print("] = increase basket_area")
    print("[ = decrease basket_area")
    print("0 = reset perception")
    print("q = quit\n")


# =========================
# FSM LOOP
# =========================
def control_loop(pub, lift_pub, claw_pub, state_pub, debug_pub):

    global block_seen, basket_seen, top_seen
    global block_error_x, basket_error_x, block_area
    global total_lift_time, basket_area

    rate = rospy.Rate(20)

    state = SEARCH_BLOCK
    last_state = None
    state_start = time.time()


    vel = Twist()
    lift = Int32()
    claw = Int32()

    Kp = 0.002
    SEARCH_SPEED = 0.3

    print_controls()

    while not rospy.is_shutdown():

        # =========================
        # STATE MACHINE
        # =========================
        debug_msg = f"block_seen={block_seen}, top_seen={top_seen}, basket_seen={basket_seen}, block_area={block_area}, basket_area={basket_area}"

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

            if block_area < 140000:
                vel.linear.x = BASE_SPEED - K_AREA * block_area
                vel.linear.x = max(MIN_SPEED, min(BASE_SPEED, vel.linear.x))
            else:
                vel.linear.x = 0.0
                vel.angular.z = 0.0
                state_start = time.time()
                state = LIFT

        elif state == LIFT:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 50
            claw.data = 0

            if time.time() - state_start > 45.0:
                state = LOWER
                total_lift_time += time.time() - state_start
                state_start = time.time()

            if top_seen:
                state = GRASP_TOP
                total_lift_time += time.time() - state_start
                state_start = time.time()

        elif state == LOWER:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = -50
            claw.data = 0

            if time.time() - state_start > 45.0:
                state = LIFT
                total_lift_time += -(time.time() - state_start)
                state_start = time.time()

            if top_seen:
                state = GRASP_TOP
                total_lift_time += -(time.time() - state_start)
                state_start = time.time()

        elif state == GRASP_TOP:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 0
            claw.data = 20

            if time.time() - state_start > 3.4:
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

            if basket_area < 100000:
                vel.linear.x = 0.08
            else:
                state_start = time.time()
                state = LIFT_BASKET


        elif state == LIFT_BASKET:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 50
            claw.data = 0

            if total_lift_time > 45.0:
                state = MOVE_TO_BASKET
                total_lift_time += time.time() - state_start
                state_start = time.time()
            else:
                state = LIFT_BASKET
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
            else:
                state = MOVE_TO_BASKET

        elif state == DROP:

            vel.linear.x = 0
            vel.angular.z = 0
            lift.data = 0
            claw.data = -20

            if time.time() - state_start > 3.0:
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
                block_seen = False
                basket_seen = False
                top_seen = False
                block_error_x = 0
                basket_error_x = 0
                block_area = 0
                basket_area = 0
                total_lift_time = 0.0
                state = SEARCH_BLOCK

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
# MAIN THREAD (KEYBOARD)
# =========================
def keyboard_loop():

    global block_seen, basket_seen, top_seen
    global block_error_x, basket_error_x, block_area, basket_area

    while True:
        key = get_key()

        if key == "q":
            print("Quitting...")
            rospy.signal_shutdown("User quit")
            break

        elif key == "b":
            block_seen = not block_seen
            print(f"block_seen = {block_seen}")

        elif key == "t":
            top_seen = not top_seen
            print(f"top_seen = {top_seen}")

        elif key == "k":
            basket_seen = not basket_seen
            print(f"basket_seen = {basket_seen}")

        elif key == "+":
            block_area += 20000
            print(f"block_area = {block_area}")

        elif key == "-":
            block_area = max(0, block_area - 20000)
            print(f"block_area = {block_area}")

        elif key == "a":
            block_error_x -= 50
            print(f"block_error_x = {block_error_x}")

        elif key == "d":
            block_error_x += 50
            print(f"block_error_x = {block_error_x}")

        elif key == "j":
            basket_error_x -= 50
            print(f"basket_error_x = {basket_error_x}")

        elif key == "l":
            basket_error_x += 50
            print(f"basket_error_x = {basket_error_x}")

        elif key == "]":
            basket_area += 20000
            print(f"basket_area = {basket_area}")

        elif key == "[":
            basket_area = max(0, basket_area - 20000)
            print(f"basket_area = {basket_area}")

        elif key == "0":
            block_seen = False
            basket_seen = False
            top_seen = False
            block_error_x = 0
            basket_error_x = 0
            block_area = 0
            basket_area = 0
            print("Reset all perception values")


def main():

    rospy.init_node("fsm_debug_keyboard")

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    lift_pub = rospy.Publisher("/lift_cmd", Int32, queue_size=10)
    claw_pub = rospy.Publisher("/claw_cmd", Int32, queue_size=10)

    state_pub = rospy.Publisher("/fsm_state", String, queue_size=10)
    debug_pub = rospy.Publisher("/fsm_debug", String, queue_size=10)

    # start FSM thread
    t = threading.Thread(target=control_loop, args=(pub, lift_pub, claw_pub, state_pub, debug_pub))
    t.daemon = True
    t.start()

    # keyboard loop in main thread
    keyboard_loop()


if __name__ == "__main__":
    main()
