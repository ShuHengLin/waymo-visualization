import os
import cv2
import numpy as np
from tqdm import tqdm

from lib.utils import *

##############################
# Options
##############################

resolution = 0.1
width = 1500
extents = np.array([[-width // 2 * resolution, width // 2 * resolution],
                    [-width // 2 * resolution, width // 2 * resolution],
                    [-2.0,   5.0]
                   ])

data_path = '/data/waymo/processed_data/train/'
data_path = '/data/waymo/processed_data/val/'

##############################
# End of Options
##############################

data_names = sorted(os.listdir(data_path + 'lidar/'))

import pickle
def load_data(load_path):
  with open(load_path, "rb") as f:
    return pickle.load(f)

# ==================================================================================================================

for data_name in tqdm(data_names):

  # loading pointcloud
  all_scan = []
  lidar_data = load_data(data_path + 'lidar/' + data_name)
  lidar_points   = lidar_data['lidars']['points_xyz']
  lidar_features = lidar_data['lidars']['points_feature']
  lidar_all_points = np.concatenate((lidar_points, lidar_features[:, 1][:, np.newaxis]), axis=1)

  pointcloud = Pointcloud(lidar_all_points)
  voxel = pointcloud.voxelize(voxel_size=(resolution, resolution, resolution),
                              extents=extents,
                              return_indices=False)
  lidar_img = np.zeros((width, width, 3)).astype(np.uint8)
  lidar_img[np.sum(voxel, axis=2) > 0] = 255.


  # loading label
  anno_data = load_data(data_path + 'annos/' + data_name)
  for obj in anno_data['objects']:
    corners = compute_box_corners(obj['box']).reshape(8, 3)
    x_col = corners[:, 0]
    y_col = corners[:, 1]
    x_col = np.floor((x_col - extents[0, 0]) / resolution).astype(np.int32)
    y_col = np.floor((y_col - extents[1, 0]) / resolution).astype(np.int32)
    corners = np.stack((y_col, x_col), axis=0)
    if obj['combined_difficulty_level'] < 2:
      draw_box_2d(lidar_img, corners, obj['label'], thickness=2)  # GT level 1 color: depends on obj class
    else:
      draw_box_2d(lidar_img, corners, cls=5, thickness=2)         # GT level 2 color: purple


  # Show result
  cv2.namedWindow ('output', 0)
  cv2.resizeWindow('output', (900, 900))
  cv2.imshow('output', lidar_img)
  cv2.waitKey(1)
