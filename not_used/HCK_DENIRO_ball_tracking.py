
# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import rospy
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

#####################################################################

# GREEN
# colorLower = (29, 86, 6)
# colorUpper = (64, 255, 255)
#BLUE
colorLower = (73, 100, 190)
colorUpper = (106, 255, 255)


# make script to detect object by clicking 
# http://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)

# grab the reference to the webcam
# camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture(1)
camera.set(cv2.cv.CV_CAP_PROP_FPS, 120)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
w = camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
h = camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
print w,h
# t = camera.get(cv2.cv.CV_CAP_PROP_FPS)
# print t

#####################################################################

def cleanup_on_shutdown():
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

#####################################################################
rospy.init_node("HCK_DENIRO_ball_tracking")

pub_ball_pos = rospy.Publisher('HCK_ball_pos', Float32MultiArray, queue_size=1)
pub_ball_img = rospy.Publisher('HCK_ball_img', Image, queue_size=1)

msg_pos = Float32MultiArray()
msg_bridge = CvBridge()

rate = rospy.Rate(240)  # 100Hz
 
#####################################################################

# keep looping
while not rospy.is_shutdown():
    # grab the current frame
    (grabbed, frame) = camera.read()
    # resize the frame, blur it, and convert it to the HSV color space
    # frame = imutils.resize(frame, width=600)
    height, width, _ = frame.shape
    # print height, width


    ### EXTRACT THE BLOB ###
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
        msg_pos.data = [center[0]-width/2.0, -(center[1]-height/2.0)]
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    else:
        msg_pos.data = [-1000, -1000] 


    ### PUBLISH the POSITION ###
    # print "X:",msg_pos.data[0],",Y:",msg_pos.data[1] 
    pub_ball_pos.publish(msg_pos)

    ### PUBLISH the IMAGE ###
    try:
        pub_ball_img.publish(msg_bridge.cv2_to_imgmsg(frame, "passthrough"))
    except CvBridgeError as e:
        print(e)

    cv2.imshow("Front Camera", frame)
    cv2.waitKey(1)

    rospy.on_shutdown(cleanup_on_shutdown)
    rate.sleep()
 