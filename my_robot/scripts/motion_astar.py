#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg  import Twist
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from tf.transformations import euler_from_quaternion

from functions import *
from math import sqrt, atan2, cos, sin, pi
from robot_astar import*

import cv2

map_origin = np.array([0.0,0.0])

def callback(odom):
    # Check the original code in this section
    # https://github.com/engcang/turtlebot2/tree/master/Python-Kinematic%20Position%20Control
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    global map

    map_data = map_config_space(map, 2)
    map_data = map_data.astype(np.uint8)
    rows = map_data.shape[0]
    cols = map_data.shape[1]
    resol = map.info.resolution
    origin_x = -int(rows + map_origin[1]/resol)
    origin_y = -int(cols + map_origin[0]/resol)

    x_w, y_w, yaw = get_pos(odom)

    x = int(origin_x + y_w/resol)
    y = int(origin_y + x_w/resol)

    traj_map, came_from = Astar(map_data, x, y, origin_x,origin_y)

    traj_list = []
    for c in came_from:
        traj_list.append([c.row, c.col])

    traj_list = np.array(traj_list)
    traj_x = (traj_list[:,1] + origin_x)*resol
    traj_y = (traj_list[:,0] + origin_y)*resol

    # print(traj_x.shape, traj_y.shape)

    mc, done = go_to_point(traj_x[0],traj_y[0],0,odom)
    print(done)
    pub.publish(mc)

    # cv2.imshow("map", traj_map)
    # cv2.imwrite("test_map2.jpg", map_data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def map_callback(map_info):
    global map
    map = map_info
    

def meta_map_callback(meta_data):
    global map_origin
    map_origin[0] = meta_data.origin.position.x
    map_origin[1] = meta_data.origin.position.y


def node_func():

    global pub
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/odom", Odometry, callback)
    rospy.Subscriber("/map", OccupancyGrid, map_callback)
    rospy.Subscriber("/map_metadata", MapMetaData, meta_map_callback)
    rospy.spin()
if __name__ == '__main__':
    node_func()
