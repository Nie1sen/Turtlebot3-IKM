import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty
import math
#part b
from sensor_msgs.msg import LaserScan

settings = termios.tcgetattr(sys.stdin)


obstacle_detected = False

# Settings for non-blocking terminal read
def get_key():
    tty.setraw(sys.stdin.fileno())
    # Wait 0.1 seconds for a keypress, otherwise return None
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

#part b
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
    rospy.init_node('turtlebot3_autonomous_move', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz
    vel_msg = Twist()

    # speed step increments
    LIN_STEP = 0.01
    ANG_STEP = 0.1

    #lidar subscriber
    rospy.Subscriber('/scan', LaserScan, scan_callback)


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
        elif key == ' ': # Spacebar to emergency stop
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
        elif key == '\x03': # Ctrl+C
            break

        if obstacle_detected and vel_msg.linear.x > 0:
            #output angle of detected obstacle
            #rospy.logwarn()
            rospy.logwarn("SAFETY STOP: Obstacle in front!")
            vel_msg.linear.x = 0.0


        pub.publish(vel_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        move()
    except rospy.ROSInterruptException:
        pass

