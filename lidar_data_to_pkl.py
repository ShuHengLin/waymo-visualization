import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tqdm import tqdm

import tensorflow as tf

from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils import lidar_utils
from waymo_open_dataset.utils import box_utils

##############################
# Options
##############################

dataset_dir = '/data/waymo/dataset/training'
output_dir  = '/data/waymo/processed_data/'

context_name = '15832924468527961_1564_160_1584_160'
#context_name = '16102220208346880_1420_000_1440_000'
#context_name = '11004685739714500220_2300_000_2320_000' 
#context_name = '10023947602400723454_1120_000_1140_000'

##############################
# End of Options
##############################

import dask.dataframe as dd
def read(tag: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  return dd.read_parquet(paths)


import pickle
def pickle_save(save_path, save_data):
  with open(save_path, "wb") as f:
    pickle.dump(save_data, f)

# ==================================================================================================================

lidar_df       = read('lidar')
lidar_box_df   = read('lidar_box')
lidar_calib_df = read('lidar_calibration')

df = v2.merge(v2.merge(lidar_df, lidar_box_df, right_group=True), lidar_calib_df, right_group=True)
progress = tqdm(total=df.shape[0].compute(), desc="Processed_lidar_data")
for i, (_, row) in enumerate(df.iterrows()):

  # Loading Component
  lidar       = v2.LiDARComponent.from_dict(row)
  lidar_calib = v2.LiDARCalibrationComponent.from_dict(row)
  lidar_box   = v2.LiDARBoxComponent.from_dict(row)
  assert lidar.key.laser_name == lidar_calib.key.laser_name
  assert lidar.key.frame_timestamp_micros == lidar_box.key.frame_timestamp_micros

  # Create output dir
  output_lidar_path = output_dir + str(lidar_calib.key.laser_name) + '/'
  if not os.path.exists(output_lidar_path):
    os.makedirs(output_lidar_path)

  output_box_path = output_dir + 'box/'
  if not os.path.exists(output_box_path):
    os.makedirs(output_box_path)

  # Convert range_image to point_cloud
  points_tensor = lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_calib)
  points = points_tensor.numpy()

  # Compute box 8 corners
  num_box = len(lidar_box.key.laser_object_id)
  corners = box_utils.get_upright_3d_box_corners(tf.stack([lidar_box.box.center.x,
                                                           lidar_box.box.center.y,
                                                           lidar_box.box.center.z,
                                                           lidar_box.box.size.x,
                                                           lidar_box.box.size.y,
                                                           lidar_box.box.size.z,
                                                           lidar_box.box.heading], axis=-1)).numpy()
  corners = corners.reshape((num_box, -1))
  box_type = np.array(lidar_box.type).reshape((num_box, 1))

  # Save
  pickle_save(output_lidar_path + str(lidar.key.frame_timestamp_micros) + '.pkl', points)
  pickle_save(output_box_path   + str(lidar.key.frame_timestamp_micros) + '.pkl', np.concatenate((box_type, corners), axis=1))
  progress.update(1)
