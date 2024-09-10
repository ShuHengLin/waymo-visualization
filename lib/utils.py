import numpy as np
from lib.Point_utils import Pointcloud

# ==================================================================================================================

# Waymo to KITTI
# https://github.com/fudan-zvg/PARTNER/blob/main/det3d/datasets/waymo/waymo_common.py

def waymo_to_kitti(obj):

  waymo_obj = np.copy(obj)

  if len(obj.shape) > 1:
    waymo_obj[:, -1] = -waymo_obj[:, -1] - np.pi / 2
    waymo_obj[:, -1] = (waymo_obj[:, -1] + np.pi) % (2 * np.pi) - np.pi
    waymo_obj[:, [3, 4]] = waymo_obj[:, [4, 3]]

  else:   
    waymo_obj[-1] = -waymo_obj[-1] - np.pi / 2
    waymo_obj[-1] = (waymo_obj[-1] + np.pi) % (2 * np.pi) - np.pi
    waymo_obj[[3, 4]] = waymo_obj[[4, 3]]

  return waymo_obj

# ==================================================================================================================

# Get_upright_3d_box_corners & get_yaw_rotation
# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/box_utils.py
# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/transform_utils.py

def compute_box_corners(box):
  """
  box: [x, y, z, l, w, h, ref_velocity[0], ref_velocity[1], heading]
  """
  yaw = box[-1]
  c, s = np.cos(yaw), np.sin(yaw)
  R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

  translation = np.stack([box[0], box[1], box[2]], axis=-1)

  l, w, h = box[3], box[4], box[5]
  l2, w2, h2 = l * 0.5, w * 0.5, h * 0.5
  corners = np.stack([ l2, w2, -h2, -l2, w2, -h2, -l2, -w2, -h2, l2, -w2, -h2, l2, w2, h2,
                      -l2, w2, h2, -l2, -w2, h2, l2, -w2, h2], axis=-1).reshape((8, 3))
  corners = np.dot(corners, R.T) + translation
  return corners

# ==================================================================================================================

# Clearing all markers / view in RVIZ remotely
# https://answers.ros.org/question/53595/clearing-all-markers-view-in-rviz-remotely/

def new_marker_array():

  from visualization_msgs.msg import Marker
  from visualization_msgs.msg import MarkerArray

  marker_array_msg = MarkerArray()
  marker = Marker()
  marker.id = 0
  marker.action = Marker.DELETEALL
  marker_array_msg.markers.append(marker)
  return marker_array_msg

# ==================================================================================================================

def marker_color_select(cls, marker):

    if cls == 1 or cls == 'VEHICLE':
      marker.color.r = 0  # Green
      marker.color.g = 1
      marker.color.b = 0

    elif cls == 2 or cls == 'PEDESTRIAN':
      marker.color.r = 1  # Red
      marker.color.g = 0
      marker.color.b = 0

    elif cls == 3 or cls == 'SIGN':
      marker.color.r = 0  # Cyan
      marker.color.g = 1
      marker.color.b = 1

    elif cls == 4 or cls == 'CYCLIST':
      marker.color.r = 1  # Yellow
      marker.color.g = 1
      marker.color.b = 0

    else:                 # Unknown
      marker.color.r = 1  # White
      marker.color.g = 1
      marker.color.b = 1

    return marker

# ==================================================================================================================

lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
         [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
         [3, 4], [0, 7]]

def box_to_marker(ob, cls, index):

  import rospy
  from geometry_msgs.msg import Point
  from visualization_msgs.msg import Marker

  detect_points_set = []
  for x in range(8):
    detect_points_set.append(Point(ob[x][0], ob[x][1], ob[x][2]))

  marker = Marker()
  marker.header.frame_id = 'map'
  marker.header.stamp = rospy.Time.now()
  marker.id = index
  marker.action = Marker.ADD
  marker.type = Marker.LINE_LIST
  marker.lifetime = rospy.Duration(0)

  marker = marker_color_select(cls, marker)
  marker.color.a = 1
  marker.scale.x = 0.2
  marker.points = []

  for line in lines:
    marker.points.append(detect_points_set[line[0]])
    marker.points.append(detect_points_set[line[1]])

  return marker

# ==================================================================================================================

def color_select(cls):

    if cls == 1 or cls == 'VEHICLE':
      color = (0, 255, 0)     # Green

    elif cls == 2 or cls == 'PEDESTRIAN':
      color = (0, 0, 255)     # Red

    elif cls == 3 or cls == 'SIGN':
      color = (255, 255, 0)   # Cyan

    elif cls == 4 or cls == 'CYCLIST':
      color = (0, 255, 255)   # Yellow

    elif cls == 5 or cls == 'Van':
      color = (255, 0, 255)   # Purple

    else:                     # Unknown
      color = (255, 255, 255) # White

    return color

# ==================================================================================================================

def draw_box_2d(image, corners, cls, thickness, score=None):

  import cv2

  c = color_select(cls)
  cv2.line(image, (corners[0][0], corners[1][0]), (corners[0][1], corners[1][1]), c, thickness, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][1], corners[1][1]), (corners[0][2], corners[1][2]), c, thickness, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][2], corners[1][2]), (corners[0][3], corners[1][3]), c, thickness, lineType=cv2.LINE_AA)
  cv2.line(image, (corners[0][3], corners[1][3]), (corners[0][0], corners[1][0]), c, thickness, lineType=cv2.LINE_AA)

  # Draw the heading of the BEV bbox (only on BEV map)
  center_x = np.mean(corners[0])
  center_y = np.mean(corners[1])
  heading_center_x = (corners[0][0] + corners[0][3]) / 2
  heading_center_y = (corners[1][0] + corners[1][3]) / 2
  cv2.line(image, (int(center_x), int(center_y)), (int(heading_center_x), int(heading_center_y)), c, thickness, lineType=cv2.LINE_AA)

  if score:
    text_x = int((np.max(corners[0]) + np.min(corners[0])) // 2) - 20
    text_y = int(np.min(corners[1])) - 5
    cv2.putText(image, text=str(round(score, 2)), org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=c, thickness=2)
  return image
