

#####################################################
# Dependencies: https://github.com/code-iai/iai_kinect2
# First run: roslaunch kinect2_bridge kinect2_bridge.launch
#####################################################

import rospy
import cv2
import pickle
import numpy as np
from imutils.video import VideoStream
import imutils

import itertools

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

#####################################################
# Blue filter
# colorLower = (75, 138, 228)
# colorUpper = (125, 255, 255)
# #
# colorLower = (73, 100, 190)
# colorUpper = (110, 255, 255)
#daylight
# colorLower = (54, 73, 255)
# colorUpper = (93, 150, 255)

# #BLUE filter
# colorLower = (85, 120, 200)       # (85, 150, 200)   
# colorUpper = (105, 255, 255)
#RED filter
# colorLowerR = (160, 80, 235)    # (125, 80, 190)
# colorUpperR = (175, 255, 255)
colorLower = (165, 80, 190)
colorUpper = (169, 200, 255)
# #GREEN filter
# colorLowerR = (70, 40, 255)
# colorUpperR = (90, 100, 255)
colorLowerR = (50, 60, 200)
colorUpperR = (90, 130, 255)
# YELLOW filter
# colorLowerR = (25, 80, 200)    # (125, 80, 190)
# colorUpperR = (50, 200, 255)
# colorLower = (30, 60, 225)
# colorUpper = (60, 255, 255)


ball_x, ball_y = None, None



FLAG = raw_input("ENTER (1) to SAVE VIDEO:\n")

if FLAG=="1":
    fname = raw_input("ENTER FILENAME:\n")
    fourcc = cv2.cv.CV_FOURCC(*"MJPG")
    (h, w) = (None, None)
    zeros = None
    writer = cv2.VideoWriter(fname+".avi", fourcc, 10, (1920, 1080), True)

#####################################################

def cleanup_on_shutdown():
    # cleanup, close any open windows
    cv2.destroyAllWindows()
    if FLAG=="1":
        writer.release()


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
    # construct a mask for the color "BLUE", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # same for RED
    maskR = cv2.inRange(hsv, colorLowerR, colorUpperR)
    maskR = cv2.erode(maskR, None, iterations=2)
    maskR = cv2.dilate(maskR, None, iterations=2)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    cntsR = cv2.findContours(maskR.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    centerR = None
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
                (255, 10, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 255), -1)
    else:
        ball_x = None
        ball_y = None


    # only proceed if at least one contour was found
    if len(cntsR) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        cR = max(cntsR, key=cv2.contourArea)
        ((xR, yR), radiusR) = cv2.minEnclosingCircle(cR)
        MR = cv2.moments(cR)
        centerR = (int(MR["m10"] / MR["m00"]), int(MR["m01"] / MR["m00"]))
        ballR_x = centerR[0]
        ballR_y = centerR[1]
        # ball_y = -(center[0]-width/2.0)
        # ball_x = center[0]-width/2.0
        # ball_y = -(center[1]-height/2.0)
        if radiusR > 1:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(xR), int(yR)), int(radiusR),
                (50, 160, 30), 2)
            cv2.circle(frame, centerR, 5, (0, 255, 255), -1)
    else:
        ballR_x = None
        ballR_y = None

##### # BLUE BALL
    overlay = frame.copy()
    cv2.rectangle(frame, (0, frame.shape[0] - 170), 
        (570, frame.shape[0]-1), (255,255,255), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    # Add text for PIXEL position
    cv2.putText(frame, "x_px_pos: {}, y_px_pos: {}".format(ball_x, ball_y),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 0), 2)
    # Add text for REAL position
    x_coord, y_coord = get_coord(ball_x, ball_y)
    cv2.putText(frame, "x_REAL: {}, y_REAL: {}".format(x_coord, y_coord),
        (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 0), 2)
    # Add text for LABELS
    angle, L = getLabels(x_coord, y_coord)
    cv2.putText(frame, "ANGLE: {}, DISTANCE: {}".format(angle, L),
        (10, frame.shape[0] - 130), cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 20, 0), 3)

##### # GREEN BALL
    overlay1 = frame.copy()
    cv2.rectangle(frame, (frame.shape[1]-570, frame.shape[0] - 170), 
        (frame.shape[1], frame.shape[0]), (255,255,255), -1)
    cv2.addWeighted(overlay1, 0.5, frame, 0.5, 0, frame)
    # Add text for PIXEL position
    cv2.putText(frame, "x_px_pos: {}, y_px_pos: {}".format(ballR_x, ballR_y),
        (frame.shape[1]-540, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 0), 2)
    # Add text for REAL position
    xR_coord, yR_coord = get_coord(ballR_x, ballR_y)
    cv2.putText(frame, "x_REAL: {}, y_REAL: {}".format(xR_coord, yR_coord),
        (frame.shape[1]-540, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 0), 2)
    # Add text for LABELS
    angleR, LR = getLabels(xR_coord, yR_coord)
    cv2.putText(frame, "ANGLE: {}, DISTANCE: {}".format(angleR, LR),
        (frame.shape[1]-540, frame.shape[0] - 130), cv2.FONT_HERSHEY_SIMPLEX,
        1, (10, 110, 10), 3)

    # PRINT LINES
    frame = frame.copy()
    cv2.line(frame, (960, 0), (960, 1080), (0, 0, 255), 2)
    # for x in xrange(-600, 601, 200):
    #     cv2.line(frame, (960 + x, 0), (960 + x, 1080), (100, 255, 0), 5)
    # for y in xrange(-400, 401, 200):
    #     cv2.line(frame, (0, 540 + y), (1920, 540 + y), (100, 255, 0), 5)
    
    # Record augmented image
    if FLAG == "1":
        writer.write(frame.copy())

    # Show augmented image
    cv2.imshow("Front Camera", frame)
    cv2.waitKey(1)


### CONVERSION
# def get_coord(puck_x, puck_y):
#     if not (puck_x and puck_y):
#         return None, None
#     px_x = puck_x
#     px_y = 1080 - puck_y
#     matx = np.abs(x_range-px_y/100.).argmin()
#     maty = np.abs(y_range-px_x/100.).argmin()
#     x_coord = Z_x[matx, maty]
#     y_coord = Z_y[matx, maty]
#     return round(x_coord,2), round(y_coord,2)

# def get_coord_centre(puck_x, puck_y):
#     if not (puck_x and puck_y):
#         return None, None
#     px_x = 1080 - puck_y
#     px_y =  - puck_x + 1920/2
#     x_coord, y_coord, _ = np.dot(H1, np.array([px_x, px_y, 1]))
#     return round(x_coord,2), round(y_coord,2)


def get_coord(puck_x, puck_y):
    if not (puck_x and puck_y):
        return None, None
    px_x = puck_x
    px_y = 1080 - puck_y

    tmp =np.array([[px_x, px_y]], np.float32)
    x_coord, y_coord = cv2.perspectiveTransform(tmp[None,:,:], mat).reshape(2,)

    return round(x_coord,2), round(y_coord,2)

# def get_coord1(puck_x, puck_y):
#     if not (puck_x and puck_y):
#         return None, None
#     px_x = puck_x
#     px_y = 1080 - puck_y
#     # x_coord, y_coord, z_coord = np.dot(H, np.array([px_x, px_y]))
#     # x_coord, y_coord, z_coord = np.dot(H1, np.array([px_x, px_y]))
#     return round(x_coord,2), round(y_coord,2), round(z_coord,2)


def getLabels(x_coord, y_coord):
    if not (x_coord and y_coord):
        return None, None
    # init_x = -22.0
    init_x = -37.0
    init_y = -76.0
    L = np.sqrt((x_coord-init_x)**2 + (y_coord-init_y)**2)
    angle = np.rad2deg(np.arctan2(y_coord-init_y, x_coord-init_x))
    return round(angle,2), round(L,2)


#####################################################################

rospy.init_node('HCK_watch_coord')
sub_ball_img = rospy.Subscriber('/kinect2/hd/image_color_rect', Image, callback=callback_cam)
# sub_ball_depth = rospy.Subscriber('/kinect2/sd/image_depth_rect', Image, callback=callback_depth)
rate = rospy.Rate(1000)

msg_bridge = CvBridge()

#####################################################################

#### GP VERSION
(Z_x, Z_y, x_range, y_range) = pickle.load( open( "kinect_mapping.dat", "rb" ) )

#### HOMOGRAPHY VERSION
# # measurements in pixels (pixel_y, pixel_x)
# m_pixels = np.array( [[0., 0.], [140., 0], [340., 0], [540., 0], [740., 0], [940., 200], 
#                     [140., 200], [340., 200], [540., 200], [740., 200], [940., 200], 
#                     [140., 400], [340., 400], [540., 400], [740., 400], [940., 400]])

# # distances in meters (robot_x, robot_y)
# m_meters = np.array( [[0., 0.], [20., 0], [56., 0], [106., 0], [180., 0], [298.,0],
#                 [20., 30], [56.5, 36], [107.5, 43], [182., 52], [298., 67],
#                 [20., 61], [57.5, 70.5], [111., 85], [187., 105], [307., 137]])

#JULY
m_pixels = np.array([[560., 740.], [1560., 940.], [1560., 140.], [960., 140.]], np.float32)
m_meters = np.array([[188., 105.], [321.,-215.], [17.,-93.], [17., 0.]], np.float32)
# MARCH
# m_pixels = np.array([[560., 740.], [1560., 940.], [1560., 140.], [960., 140.]], np.float32)
# m_meters = np.array([[188., 105.], [314.,-203.], [20.,-92.], [20., 0.]], np.float32)
# OLD
# m_pixels = np.array([[960., 140., 1], [560., 740., 1], [1560., 140., 1], [1560., 940., 1]])
# m_meters = np.array([[20., 0., 1],[188., 105., 1],[20.,-92., 1],[314.,-203., 1]])

mat = cv2.getPerspectiveTransform(m_pixels, m_meters)

# H, status = cv2.findHomography(m_pixels, m_meters, method=0)
# H1, status = cv2.findHomography(m_pixels, m_meters, cv2.RANSAC, ransacReprojThreshold=1.0)

# H1, status = cv2.findHomography(m_px_centre, m_meters, method=0)
# H, status = cv2.findHomography(x_train, y_train, method=0, ransacReprojThreshold=3.0)

while not rospy.is_shutdown():
    # Loop
    pass

rospy.on_shutdown(cleanup_on_shutdown)
rospy.spin()

