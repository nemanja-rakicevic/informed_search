

######################################################################

- In order to use the Kinect with ROS you need to install
  iai_kinect2 bridge from https://github.com/code-iai/iai_kinect2
  and all the necessary dependencies.

- hck_cam2xy.py
  Gets the rectified image from the Kinect and does object
  tracking for specified colors.

- util_RGB_range.py
  Script used to extract the color ranges of the desired objects manually
  usage: 
  $ python util_RGB_range.py -w -p -f HSV 

- kinect_mapping.dat
  Transformation data used to transform the image coordinate to the
  floor coordinates.

- test_kinect.py
  Just some untested code to play around.
  
######################################################################