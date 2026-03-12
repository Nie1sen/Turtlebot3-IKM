import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import sys, select, termios, tty
import math

settings = termios.tcgetattr(sys.stdin)

# Safety threshold
SAFETY_DISTANCE = 0.3
obstacle_detected = False


# LaserScan callback
def scan_callback(msg):
    global obstacle_detected

    # Check objects directly in front (small window around center)
    center = len(msg.ranges) // 2
    window = 10

    front_ranges = msg.ranges[center-window:center+window]

    obstacle_detected = False

    for r in front_ranges:
        if not math.isinf(r) and r < SAFETY_DISTANCE:
            obstacle_detected = True
            break


# Settings for non-blocking terminal read
def get_key():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def move():
    global obstacle_detected

    rospy.init_node('turtlebot3_autonomous_move', anonymous=True)

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Subscriber to LiDAR
    rospy.Subscriber('/scan', LaserScan, scan_callback)

    rate = rospy.Rate(10)
    vel_msg = Twist()

    LIN_STEP = 0.01
    ANG_STEP = 0.1

    while not rospy.is_shutdown():

        key = get_key()

        if key == 'w':
            vel_msg.linear.x += LIN_STEP
        elif key == 's':
            vel_msg.linear.x -= LIN_STEP
        elif key == 'a':
            vel_msg.angular.z += ANG_STEP
        elif key == 'd':
            vel_msg.angular.z -= ANG_STEP
        elif key == ' ':
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
        elif key == '\x03':
            break

        # Safety override
        if obstacle_detected:
            if vel_msg.linear.x > 0:
                vel_msg.linear.x = 0.0

        pub.publish(vel_msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        move()
    except rospy.ROSInterruptException:
        pass
