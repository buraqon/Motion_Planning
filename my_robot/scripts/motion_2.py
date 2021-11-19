#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg  import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from tf.transformations import euler_from_quaternion

from functions import *
from math import sqrt, atan2, cos, sin, pi

import cv2


def callback(odom):
    # Check the original code in this section
    # https://github.com/engcang/turtlebot2/tree/master/Python-Kinematic%20Position%20Control
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    #mc,done = go_to_point(0, 0, 0, odom)
    #print(done)
    #pub.publish(mc)

def map_config_space(map, offset = 3):
    map_data = np.array(map.data)
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

def map_callback(map):
    
    map_data = map_config_space(map, 2)

    map_data = map_data.astype(np.uint8)
    cv2.imshow("map", map_data)
    cv2.imwrite("test_map.jpg", map_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def node_func():

    global pub
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/odom", Odometry, callback)
    rospy.Subscriber("/map", OccupancyGrid, map_callback)
    rospy.spin()
if __name__ == '__main__':
    node_func()
