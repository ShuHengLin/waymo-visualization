import os
import rospy
import numpy as np
from tqdm import tqdm

from lib.utils_pointcloud import *

import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

##############################
# Options
##############################

data_path = '/data_1TB_1/waymo/processed_data/'

##############################
# End of Options
##############################

data_names = sorted(os.listdir(data_path + str(1)))

# ==================================================================================================================

header = std_msgs.msg.Header()
header.frame_id = 'map'

pointcloud_pub = rospy.Publisher('/pointcloud',   PointCloud2, queue_size=10)
marker_pub     = rospy.Publisher('/detect_box3d', MarkerArray, queue_size=10)
rospy.init_node('talker', anonymous=True)
rate = rospy.Rate(1000)

# ==================================================================================================================

for data_name in tqdm(data_names):
  
  if rospy.is_shutdown():
    break

  # loading pointcloud
  all_scan = []
  for i in range(5):
    scan = np.load(data_path + str(i + 1) + '/' + data_name)
    all_scan.append(scan)

  all_points = np.concatenate(all_scan, axis=0)
  pointcloud_msg = pcl2.create_cloud_xyz32(header, all_points[:, 0:3])
  pointcloud_pub.publish(pointcloud_msg)


  # loading label
  marker_array = new_marker_array()
  boxes = np.load(data_path + 'box/' + data_name)
  for i, box in enumerate(boxes):
    marker = box_to_marker(box[1:].reshape(8, 3), cls=box[0], index=i)
    marker_array.markers.append(marker)
  marker_pub.publish(marker_array)

  rate.sleep()
