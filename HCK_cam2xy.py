

####################################################
# First run: roslaunch kinect2_bridge kinect2_bridge.launch

import rospy
import cv2
import pickle
import numpy as np

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

#####################################################
# Blue filter
# colorLower = (75, 138, 228)
# colorUpper = (125, 255, 255)
#
colorLower = (73, 100, 190)
colorUpper = (106, 255, 255)
#daylight
# colorLower = (54, 73, 255)
# colorUpper = (93, 150, 255)

ball_x, ball_y = None, None

#####################################################

def cleanup_on_shutdown():
    # cleanup, close any open windows
    cv2.destroyAllWindows()


def callback_cam(msg):
    global ball_x, ball_y
    # Read image
    try:
        frame = msg_bridge.imgmsg_to_cv2(msg)
    except CvBridgeError as e:
        print(e)
    # Extract blob position
    height, width, _ = frame.shape
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        ball_x = center[0]
        ball_y = center[1]
        # ball_y = -(center[0]-width/2.0)
        # ball_x = center[0]-width/2.0
        # ball_y = -(center[1]-height/2.0)
        if radius > 1:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    else:
        ball_x = None
        ball_y = None
    # Add text for PIXEL position
    cv2.putText(frame, "x_px_pos: {}, y_px_pos: {}".format(ball_x, ball_y),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 20, 255), 2)
    # Add text for REAL position
    x_coord, y_coord = get_coord(ball_x, ball_y)
    cv2.putText(frame, "x_REAL: {}, y_REAL: {}".format(x_coord, y_coord),
        (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 20, 255), 2)

    # Add text for LABELS
    angle, L = getLabels(x_coord, y_coord)
    cv2.putText(frame, "ANGLE: {}, DISTANCE: {}".format(angle, L),
        (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 20, 0), 3)

    # Show augmented image
    cv2.imshow("Front Camera", frame)
    cv2.waitKey(1)


### CONVERSION
def get_coord(puck_x, puck_y):
    if not (puck_x and puck_y):
        return None, None
    px_x = puck_x
    px_y = 1080 - puck_y
    matx = np.abs(x_range-px_y/100.).argmin()
    maty = np.abs(y_range-px_x/100.).argmin()
    x_coord = Z_x[matx, maty]
    y_coord = Z_y[matx, maty]
    return round(x_coord,2), round(y_coord,2)


def getLabels(x_coord, y_coord):
    if not (x_coord and y_coord):
        return None, None
    init_x = -15
    init_y = -15
    L = np.sqrt((x_coord-init_x)**2 + (y_coord-init_y)**2)
    angle = np.rad2deg(np.arctan2(y_coord-init_y, x_coord-init_x))
    return round(angle,2), round(L,2)


#####################################################################

rospy.init_node('HCK_watch_coord')
sub_ball_img = rospy.Subscriber('/kinect2/hd/image_color', Image, callback=callback_cam)
# sub_ball_depth = rospy.Subscriber('/kinect2/sd/image_depth_rect', Image, callback=callback_depth)
rate = rospy.Rate(100)

msg_bridge = CvBridge()

#####################################################################

(Z_x, Z_y, x_range, y_range) = pickle.load( open( "kinect_mapping.dat", "rb" ) )

while not rospy.is_shutdown():
    # Loop
    pass

rospy.on_shutdown(cleanup_on_shutdown)
rospy.spin()

