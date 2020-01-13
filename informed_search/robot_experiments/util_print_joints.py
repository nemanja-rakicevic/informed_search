#!/usr/bin/env python

import rospy
import baxter_interface as BI
import ik_solver

#####################################################################
rospy.init_node('test')

limb_left = BI.Limb("left")
limb_right = BI.Limb("right")

pose_tmp_left = limb_left.endpoint_pose()
pose_tmp_right = limb_right.endpoint_pose()
joint_values_left = ik_solver.ik_solve('left',
                                       pose_tmp_left['position'],
                                       pose_tmp_left['orientation'],
                                       limb_left.joint_angles())
joint_values_right = ik_solver.ik_solve('right',
                                        pose_tmp_right['position'],
                                        pose_tmp_right['orientation'],
                                        limb_right.joint_angles())
print '\nLEFT:\n'
print joint_values_left
print '\nRIGHT:\n'
print joint_values_right

exit()
