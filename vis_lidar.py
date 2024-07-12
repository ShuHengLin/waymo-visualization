import os
import rospy
import numpy as np
from tqdm import tqdm

from lib.utils_pointcloud import *

import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

##############################
# Options
##############################

data_path = '/data_1TB_1/waymo/processed_data/train/'

##############################
# End of Options
##############################

data_names = sorted(os.listdir(data_path + 'lidar/'))

import pickle
def load_data(load_path):
  with open(load_path, "rb") as f:
    return pickle.load(f)

# ==================================================================================================================

header = std_msgs.msg.Header()
header.frame_id = 'map'

fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
          PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
          PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
          PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)]

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
  lidar_data = load_data(data_path + 'lidar/' + data_name)
  lidar_points   = lidar_data['lidars']['points_xyz']
  lidar_features = lidar_data['lidars']['points_feature']
  lidar_all_points = np.concatenate((lidar_points, lidar_features[:, 1][:, np.newaxis]), axis=1)

  pointcloud_msg = pcl2.create_cloud(header, fields, lidar_all_points)
  pointcloud_pub.publish(pointcloud_msg)


  # loading label
  marker_array = new_marker_array()
  anno_data = load_data(data_path + 'annos/' + data_name)

  for i, obj in enumerate(anno_data['objects']):
    corners = compute_box_corners(obj['box'])
    marker = box_to_marker(corners.reshape(8, 3), cls=obj['label'], index=i)
    marker_array.markers.append(marker)
  marker_pub.publish(marker_array)

  rate.sleep()
