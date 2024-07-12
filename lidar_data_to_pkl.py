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

##############################
# Options
##############################

dataset_dir = '/data/waymo/dataset/training'
output_dir  = '/data/waymo/processed_data/train/'

#context_name = '15832924468527961_1564_160_1584_160'
#context_name = '16102220208346880_1420_000_1440_000'
context_name = '1005081002024129653_5313_150_5333_150'

##############################
# End of Options
##############################

import dask.dataframe as dd
def read(tag: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  return dd.read_parquet(paths)


def create_dir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)


import pickle
def pickle_save(save_path, save_data):
  with open(save_path, "wb") as f:
    pickle.dump(save_data, f)

# ==================================================================================================================

lidar_df        = read('lidar')
lidar_pose_df   = read('lidar_pose')
lidar_box_df    = read('lidar_box')
lidar_calib_df  = read('lidar_calibration')
vehicle_pose_df = read('vehicle_pose')

pose_df = v2.merge(lidar_pose_df, vehicle_pose_df)
df = v2.merge(v2.merge(v2.merge(pose_df, lidar_df), lidar_calib_df), lidar_box_df, right_group=True)
num_frames = int(df.shape[0].compute())
progress = tqdm(total=num_frames, desc="Processed_lidar_data")

for idx, (_, row) in enumerate(df.iterrows()):

  # Loading Component
  lidar        = v2.LiDARComponent.from_dict(row)
  lidar_pose   = v2.LiDARPoseComponent.from_dict(row)
  lidar_box    = v2.LiDARBoxComponent.from_dict(row)
  lidar_calib  = v2.LiDARCalibrationComponent.from_dict(row)
  vehicle_pose = v2.VehiclePoseComponent.from_dict(row)


  # Convert range_image to point_cloud
  points_tensor1 = lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_calib,
                                                                  lidar_pose.range_image_return1, vehicle_pose, keep_polar_features=True)
  points_tensor2 = lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return2, lidar_calib,
                                                                  lidar_pose.range_image_return1, vehicle_pose, keep_polar_features=True)
  points_tensor = tf.concat([points_tensor1, points_tensor2], axis=0)


  # Create output dir
  output_lidar_path = output_dir + 'lidar/'
  output_anno_path  = output_dir + 'annos/'
  create_dir(output_lidar_path)
  create_dir(output_anno_path)


  # Save lidar
  lidars = {'points_xyz'    : points_tensor[:, 3:].numpy(),
            'points_feature': points_tensor[:, 1:3].numpy()
           }
  lidar_data = {'scene_name': lidar.key.segment_context_name,
                'frame_name': '{scene_name}_{timestamp}'.format(scene_name=lidar.key.segment_context_name,
                                                                timestamp=lidar.key.frame_timestamp_micros),
                'frame_id'  : idx,
                'lidars'    : lidars
               }
  pickle_save(output_lidar_path + 'seq_0_frame_' + str(idx) + '.pkl', lidar_data)


  # Save annos
  objects = []
  for i in range(len(lidar_box.key.laser_object_id)):
    objects.append({'id'                        : i,
                    'name'                      : lidar_box.key.laser_object_id[i],
                    'label'                     : lidar_box.type[i],
                    'box'                       : np.array([lidar_box.box.center.x[i], lidar_box.box.center.y[i], lidar_box.box.center.z[i],
                                                            lidar_box.box.size.x[i], lidar_box.box.size.y[i], lidar_box.box.size.z[i], 0, 0,
                                                            lidar_box.box.heading[i]], dtype=np.float32),
                    'num_points'                : lidar_box.num_lidar_points_in_box[i],
                    'detection_difficulty_level': lidar_box.difficulty_level.detection[i],
                    'combined_difficulty_level' : 0,
                    'global_speed'              : [lidar_box.speed.x[i], lidar_box.speed.y[i]],
                    'global_accel'              : [lidar_box.acceleration.x[i], lidar_box.acceleration.y[i]]
                    })

  annos = {'scene_name'   : lidar.key.segment_context_name,
           'frame_name'   : '{scene_name}_{timestamp}'.format(scene_name=lidar.key.segment_context_name,
                                                              timestamp=lidar.key.frame_timestamp_micros),
           'frame_id'     : idx,
           'veh_to_global': vehicle_pose.world_from_vehicle.transform,  
           'objects'      : objects
          }
  pickle_save(output_anno_path + 'seq_0_frame_' + str(idx) + '.pkl', annos)
  progress.update(1)
