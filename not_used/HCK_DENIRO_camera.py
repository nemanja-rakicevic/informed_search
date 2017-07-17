
#####################################################################
# used to broadcast webcam image as rostopic


import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

#####################################################################

# grab the reference to the webcam
# camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture(0)
camera.set(cv2.cv.CV_CAP_PROP_FPS, 120)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
# w = camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
# h = camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
# print w,h
# t = camera.get(cv2.cv.CV_CAP_PROP_FPS)
# print t

#####################################################################

def cleanup_on_shutdown():
    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

#####################################################################
rospy.init_node("HCK_DENIRO_camera")
pub_ball_img = rospy.Publisher('HCK_ball_img', Image, queue_size=1)
msg_bridge = CvBridge()

rate = rospy.Rate(240)  # 100Hz
 
#####################################################################

# keep looping
while not rospy.is_shutdown():
    # Grab the current frame
    (grabbed, frame) = camera.read()
    # Publish the image
    try:
        pub_ball_img.publish(msg_bridge.cv2_to_imgmsg(frame, "passthrough"))
    except CvBridgeError as e:
        print(e)

    rospy.on_shutdown(cleanup_on_shutdown)
    rate.sleep()
 