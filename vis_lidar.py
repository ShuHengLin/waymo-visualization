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

data_path = '/data_1TB_1/waymo/processed_data/'

##############################
# End of Options
##############################

data_names = sorted(os.listdir(data_path + str(1)))

import pickle
def load_data(load_path):

  if load_path.endswith('.pkl'):
    with open(load_path, "rb") as f:
      return pickle.load(f)

  elif load_path.endswith('.npy'):
      return np.load(load_path)

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
  for i in range(5):
    input_points = load_data(data_path + str(i + 1) + '/' + data_name)
    scan = input_points[:, 3:]
    intensity = input_points[:, 2].reshape(-1, 1)
    all_scan.append(np.concatenate((scan, intensity), axis=1))

  all_points = np.concatenate(all_scan, axis=0)
  pointcloud_msg = pcl2.create_cloud(header, fields, all_points)
  pointcloud_pub.publish(pointcloud_msg)


  # loading label
  marker_array = new_marker_array()
  boxes = load_data(data_path + 'box/' + data_name)

  for i, box in enumerate(boxes):
    marker = box_to_marker(box[1:].reshape(8, 3), cls=box[0], index=i)
    marker_array.markers.append(marker)
  marker_pub.publish(marker_array)

  rate.sleep()
