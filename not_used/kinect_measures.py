#! /usr/bin/python

#--------------------------------#
# Kinect v2 point cloud visualization using a Numpy based 
# real-world coordinate processing algorithm and OpenGL.
#--------------------------------#

import sys
import numpy as np
import cv2

#from pyqtgraph.Qt import QtCore, QtGui
#import pyqtgraph.opengl as gl

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Frame, libfreenect2

import matplotlib.pyplot as plt

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial)

types = 0
types |= FrameType.Color
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)

device.start()

def extract_coords_puck(frame):
    colorLower = (73, 100, 190)
    colorUpper = (106, 255, 255)

    #global ball_x, ball_y
    # Read image
    
    # Extract blob position
    #frame = frame.asarray()

    #height, width = 424, 512
    #frame = frame.reshape(height, width, 3)

    #cv2.imshow("Front Camera", (frame*255).astype('uint8')) #(frame * 255).astype(int))
    #cv2.waitKey(1)

    #print frame[:,:,0].min(), frame[:,:,0].mean(), frame[:,:,0].max()
    #plt.imshow(frame[:,:,0])
    #plt.show()

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
            frame = frame.copy()
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    else:
        ball_x = None
        ball_y = None

    print frame.shape
    frame = frame.copy()
    for x in xrange(-600, 601, 200):
        cv2.line(frame, (960 + x, 0), (960 + x, 1080), (0, 255, 255), 5)
    for y in xrange(-400, 401, 200):
        cv2.line(frame, (0, 540 + y), (1920, 540 + y), (0, 255, 255), 5)
    cv2.imshow("Front Camera", frame) #(frame * 255).astype(int))
    cv2.waitKey(1)
    
    return ball_x, ball_y

if __name__ == '__main__':
    while True:
        frames = listener.waitForNewFrame()
        #ir = frames["ir"]
        color = frames["color"]
        #depth = frames["depth"]

        color = color.asarray()
        color = color[:,::-1,:]
        print extract_coords_puck(color)

        listener.release(frames)

device.stop()
device.close()

sys.exit(0)
