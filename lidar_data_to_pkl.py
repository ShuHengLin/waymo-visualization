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

from multiprocessing import Pool 

##############################
# Options
##############################

dataset_dir = '/data/waymo/dataset/training/'
output_dir  = '/data/waymo/processed_data/train/'

#dataset_dir = '/data/waymo/dataset/validation/'
#output_dir  = '/data/waymo/processed_data/val/'

##############################
# End of Options
##############################

fnames = None
progress = None

output_lidar_path = output_dir + 'lidar/'
output_anno_path  = output_dir + 'annos/'

import dask.dataframe as dd
def read(tag: str, context_name: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  global fnames
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}')
  return dd.read_parquet(paths)


def create_dir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)


import pickle
def pickle_save(save_path, save_data):
  with open(save_path, "wb") as f:
    pickle.dump(save_data, f)

# ==================================================================================================================

def convert(idx):
  global fnames, progress, output_lidar_path, output_anno_path
  fname = fnames[idx]

  if fname.endswith('.parquet'):
    lidar_df        = read('lidar',             fname)
    lidar_pose_df   = read('lidar_pose',        fname)
    lidar_box_df    = read('lidar_box',         fname)
    lidar_calib_df  = read('lidar_calibration', fname)
    vehicle_pose_df = read('vehicle_pose',      fname)

    pose_df = v2.merge(lidar_pose_df, vehicle_pose_df)
    df = v2.merge(v2.merge(v2.merge(pose_df, lidar_df), lidar_calib_df), lidar_box_df, right_group=True)

    for frame_idx, (_, row) in enumerate(df.iterrows()):

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
      pickle_save(output_lidar_path + 'seq_' + str(idx) + '_frame_' + str(frame_idx) + '.pkl', lidar_data)


      # Save annos
      objects = []
      for i in range(len(lidar_box.key.laser_object_id)):

        # Difficulty level is 0 if labeler did not say this was LEVEL_2.
        # Set difficulty level of "999" for boxes with no points in box.
        if lidar_box.num_lidar_points_in_box[i] <= 0:
          combined_difficulty_level = 999

        if lidar_box.difficulty_level.detection[i] == 0:
          # Use points in box to compute difficulty level.
          if lidar_box.num_lidar_points_in_box[i] >= 5:
            combined_difficulty_level = 1
          else:
            combined_difficulty_level = 2
        else:
          combined_difficulty_level = lidar_box.num_lidar_points_in_box[i]

        objects.append({'id'                        : i,
                        'name'                      : lidar_box.key.laser_object_id[i],
                        'label'                     : lidar_box.type[i],
                        'box'                       : np.array([lidar_box.box.center.x[i], lidar_box.box.center.y[i], lidar_box.box.center.z[i],
                                                                lidar_box.box.size.x[i], lidar_box.box.size.y[i], lidar_box.box.size.z[i], 0, 0,
                                                                lidar_box.box.heading[i]], dtype=np.float32),
                        'num_points'                : lidar_box.num_lidar_points_in_box[i],
                        'detection_difficulty_level': lidar_box.difficulty_level.detection[i],
                        'combined_difficulty_level' : combined_difficulty_level,
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
      pickle_save(output_anno_path + 'seq_' + str(idx) + '_frame_' + str(frame_idx) + '.pkl', annos)

    progress.update(1)

# ==================================================================================================================

def main():
  global fnames, progress, output_lidar_path, output_anno_path

  fnames = sorted(os.listdir(dataset_dir + 'lidar/')) #[0:1]
  print("Number of files: {}".format(len(fnames)))

  # Create output dir
  create_dir(output_lidar_path)
  create_dir(output_anno_path)

  progress = tqdm(total=len(fnames), desc="Processed_lidar_data")

  with Pool(2) as p: # change according to your cpu & gpu
    r = list(p.imap(convert, range(len(fnames))))


if __name__ == '__main__':
  main()
