

import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    # Point,
    # Quaternion,
)
from std_msgs.msg import Header

from sensor_msgs.msg import JointState

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)


def ik_solve(limb, pos, orient, curr_jnt_angles):
    # ~ rospy.init_node("rsdk_ik_service_client")
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    # print "iksvc: ", iksvc
    # print "ikreq: ", ikreq
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {str(limb): PoseStamped(header=hdr,
                                    pose=Pose(position=pos,
                                              orientation=orient))}
    ikreq.pose_stamp.append(poses[limb])

    seed_msg = JointState()
    seed_msg.header.stamp = rospy.Time.now()
    seed_msg.name = curr_jnt_angles.keys()
    seed_msg.position = curr_jnt_angles.values()
    ikreq.seed_angles.append(seed_msg)

    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1
    if (resp.isValid[0]):
        # print("SUCCESS - Valid Joint Solution Found:")
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        # print limb_joints
        return limb_joints
    else:
        print("INVALID POSE - No Valid Joint Solution Found.")

    return -1
