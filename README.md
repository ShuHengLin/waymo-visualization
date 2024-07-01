# waymo-visualization

## Prepare Data
* [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install#linux)
```
cd /data_1TB_1/waymo/dataset/  # Your data path
```
```
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-466.0.0-linux-x86_64.tar.gz
```
```
tar -xf google-cloud-cli-466.0.0-linux-x86_64.tar.gz
```
```
./google-cloud-sdk/install.sh
```

* Download training set: 
```
./google-cloud-sdk/bin/gsutil -m cp -r \
"gs://waymo_open_dataset_v_2_0_0/training/lidar" \
"gs://waymo_open_dataset_v_2_0_0/training/lidar_box" \
"gs://waymo_open_dataset_v_2_0_0/training/lidar_calibration" \
/data_1TB_1/waymo/dataset/training/
```

* Download validation set: 
```
./google-cloud-sdk/bin/gsutil -m cp -r \
"gs://waymo_open_dataset_v_2_0_0/validation/lidar" \
"gs://waymo_open_dataset_v_2_0_0/validation/lidar_box" \
"gs://waymo_open_dataset_v_2_0_0/validation/lidar_calibration" \
/data_1TB_1/waymo/dataset/validation/
```


## Docker
```
xhost +local:

docker run -itd \
--name docker_Waymo \
-e DISPLAY=$DISPLAY \
-e QT_X11_NO_MITSHM=1 \
-v "/tmp/.X11-unix:/tmp/.X11-unix" \
-v /data_1TB_1/:/data/ \
--gpus all \
--shm-size=64g \
--privileged \
--ipc=host \
--network=host \
nvcr.io/nvidia/pytorch:23.07-py3 \
bash && docker exec -it docker_Waymo bash
```
```
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install protobuf-compiler ffmpeg python3-tk libsm6 libxext6 libxrender1 -y
```
```
python3 -m pip install gcsfs waymo-open-dataset-tf-2-11-0==1.6.1
```


## Visualize lidar pointcloud
```
cd /data/waymo/ && python -B lidar_data_to_npy.py
```
```
roscore
```
```
rosrun rviz rviz -d rviz_config.rviz
```
```
python -B vis_lidar.py
```


## Video


## References
1) [[2020 CVPR] Scalability in Perception for Autonomous Driving: Waymo Open Dataset](https://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.html)
2) [Waymo Open Dataset Labeling Specifications](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/labeling_specifications.md)
3) [Download Waymo Open Dataset](https://waymo.com/open/download/)
