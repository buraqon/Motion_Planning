#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg  import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

from math import sqrt, atan2, cos, sin, pi

# Get Odometry Pose
def get_pos(odom):
    odom_info = odom.pose.pose
    x = odom_info.position.x
    y = odom_info.position.y

    orient = odom_info.orientation
    (roll, pitch, yaw) = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

    return x, y, yaw


def go_to_point(xt,yt,yawt,odom):
    x, y, yaw = get_pos(odom)

    K1 = 0.5
    K2 = 0.5

    dthresh = 0.1
    athresh = 0.1

    r = sqrt((xt - x)**2 + (yt-y)**2)
    psi = atan2(yt-y, xt-x)

    phi = yaw - psi
    if phi > pi:
        phi = phi - 2*pi
    if phi < -pi:
        phi = phi + 2*pi

    if (r >= dthresh):
        V = K1*r*cos(phi)
        w = -K1*sin(phi)*cos(phi) - (K2*phi)
        done = 0
    else:
        V = 0
        if (abs(yawt - yaw) >= athresh):
            w = K2*(yawt - yaw)
            done = 0
        else:
            w = 0
            done = 1


    mc = Twist()
    mc.linear.x = V
    mc.angular.z = w

    # Return the movement command and a boolean to check if the order is done
    return mc, done

def map_config_space(map, offset = 3):
    map_data = np.array(map.data)
    #print(map.info.resolution)
    (width, height) = (map.info.width, map.info.height)
    map_data = np.reshape(map_data, (height, width))
    map_data = np.where(map_data >= 0, map_data, 50)

    offset_arr = np.array([[1, 1],
                           [1, -1],
                           [-1, 1],
                           [-1, -1]])

    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            if(map_data[i][j] == 100):
                for ofs in offset_arr:
                    for k in range(offset):
                        ofx = (k+1)*ofs[0]
                        ofy = (k+1)*ofs[1]
                        if(i+ofx < height and j+ofy < width):
                            map_data[i+ofx][j+ofy] = 99
    return map_data


