import numpy as np
import gzip
import json
import os
from tqdm import tqdm
from name_mapping import NameMapping
import laspy
import pandas as pd
from copy import deepcopy
import logging
from functools import partial
from pqdm.processes import pqdm
import argparse

ego2lidar = np.array([[0.0013955, -0.9999983, -0.0012190, 0],
     [0.9999865,  0.0013894,  0.0050018, 0.49],
     [-0.0050001, -0.0012259,  0.9999868, -2.0565],
     [0, 0, 0, 1]])
ue2righthead = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
cam_template = {"intrinsics":{
    "RadialDistortion" : [-1, 0, 0],
    "TangentialDistortion": [0, 0],
    "ImageSize": [1080,1920],
    "IntrinsicMatrix": None,
    },
    "extrinsics":{"cTv": None}}

index_labels = [
    'timestamp_nanoseconds', 'middle_lidar',
    'front_right_camera', 'front_left_camera', 'left_camera', 
    'back_left_camera', 'back_right_camera', 'right_camera'
]

FOLDER_NANEM_MAP = {'rgb_left':'left_camera',
                    'rgb_front_left':'front_left_camera',
                    'rgb_front_right':'front_right_camera',
                    'rgb_right':'right_camera',
                    'rgb_back_left':'back_left_camera',
                    'rgb_back_right':'back_right_camera'}

NUSCENES_2_OBJ_CATEGORIES = {
    "bicycle": "Cyclist",
    "car": "Car",
    "van": "Van",
    "truck": "Truck",
    "bus": "Bus",
    "traffic_cone": "Other",
    "motorcycle": "Motorcycle",
    "pedestrian": "Pedestrian",
    "others": "Other"}

CAMERA_TO_FOLDER_MAP = {'CAM_LEFT':'left_camera',
                        'CAM_FRONT_LEFT':'front_left_camera',
                        'CAM_FRONT_RIGHT':'front_right_camera',
                        'CAM_RIGHT':'right_camera',
                        'CAM_BACK_LEFT':'back_left_camera',
                        'CAM_BACK_RIGHT':'back_right_camera'}

CARLA_CHILD_ID = {9, 10, 11, 12, 13, 14, 48, 49}
CARLA_CLS_EMERGENCY = {"vehicle.ford.ambulance",
                       "vehicle.carlamotors.firetruck",
                       "vehicle.dodge.charger_police_2020",
                       "vehicle.dodge.charger_police",}
CARLA_CLS2ATTR = {"Bus": {"6": "rigid"},
                  "Truck": {"2": "regular"},
                  "Car": {"2": "regular"},
                  "Van": {"2": "regular"},
                  "Motorcycle": {"5": "r"},
                  "Cyclist": {"5": "r"}}




EDGES = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
MIN_LIDAR_POINTS_THRESHOLD = 5
PROJ_LIDAR2EGO = False

def invert_pose(pose):
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = np.transpose(pose[:3, :3])
    inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
    return inv_pose

def compute_obb_location_and_yaw_ordered(vertices):
    location = np.mean(vertices, axis=0)
    face_plus_x_center = np.mean(vertices[0:4], axis=0)
    face_minus_x_center = np.mean(vertices[4:8], axis=0)
    x_axis_vector = face_plus_x_center - face_minus_x_center
    yaw = np.arctan2(x_axis_vector[1], x_axis_vector[0])
    return location, yaw

def name2cls_attr(carla_name):
    if "walker" in carla_name:
        cls_id = int(carla_name.split(".")[-1])
        if cls_id in CARLA_CHILD_ID:
            return "Pedestrian", {"1": "child", "4": "w"}
        else:
            return "Pedestrian", {"1": "adult", "4": "w"}
    elif carla_name in NameMapping:
        obj_cls = NUSCENES_2_OBJ_CATEGORIES[NameMapping[carla_name]]
        if carla_name in CARLA_CLS_EMERGENCY:
            return obj_cls, {"2": "emergency"}
        elif obj_cls in CARLA_CLS2ATTR:
            attr_dict = CARLA_CLS2ATTR[obj_cls]
            return obj_cls, attr_dict
        else:
            return obj_cls, {}
    else:
        return "Other", {}
    
def get_calib_json(anno_dir, save_path):
    anno_files = sorted(os.listdir(anno_dir))
    anno_file = os.path.join(anno_dir, anno_files[0])

    with gzip.open(anno_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)

    calibration = {}
    if not PROJ_LIDAR2EGO:
        lidar_params = {"extrinsics":{"vTl": np.linalg.inv(ego2lidar).tolist()}}
    else:
        lidar_params = {"extrinsics":{"vTl": np.eye(4).tolist()}}
    adma_params = {"extrinsics":{"vTa": [
        [1.0, 0.0, 0.0, -0.303908],
        [0.0, 1.0, 0.0, 0.026288],
        [0.0, 0.0, 1.0, -0.28586],
        [0.0, 0.0, 0.0, 1.0]]}}
    calibration = {"middle_lidar": lidar_params, "adma": adma_params}

    world2ego = data["bounding_boxes"][0]["world2ego"]
    ego_extent = np.array(data["bounding_boxes"][0]["extent"]) * 2
    calibration["state"] = {"dimension": ego_extent.tolist()}

    for sensor_name, sensor_params in data["sensors"].items():
        if "CAM" in sensor_name:
            intrinsic = np.array(sensor_params["intrinsic"]).T
            ego2cam = np.array(sensor_params["cam2ego"])
            
            ego2cam = invert_pose(ego2cam)
            ego2cam = np.dot(ue2righthead, ego2cam)
            
            curr_cam_param = deepcopy(cam_template)
            curr_cam_param["intrinsics"]["IntrinsicMatrix"] = intrinsic.tolist()
            curr_cam_param["extrinsics"]["cTv"] = ego2cam.tolist()
            calibration[CAMERA_TO_FOLDER_MAP[sensor_name]] = curr_cam_param

    with open(f"{save_path}/calibration.json", 'w') as f:
        json.dump(calibration, f, indent=4)

def create_timesync_file(work_dir, save_path):
    lidar_files = sorted(os.listdir(f"{work_dir}/lidar"))
    pd_dict = {}
    for i in range(0, len(lidar_files), 2):
        curr_timestamp = int((i / 20) * 1e9)
        pd_index = 1 + i // 2
        pd_dict[pd_index] = [curr_timestamp,
                             f"{curr_timestamp}.npz",
                             f"{i:05d}.jpg",
                             f"{i:05d}.jpg",
                             f"{i:05d}.jpg",
                             f"{i:05d}.jpg",
                             f"{i:05d}.jpg",
                             f"{i:05d}.jpg"]
    pd_df = pd.DataFrame.from_dict(pd_dict, orient='index', columns=index_labels)
    pd_df = pd_df.transpose()
    pd_df.to_csv(f"{save_path}/timesync_info.csv")
    

def preprocess_anno(anno_dir):
    anno_files = sorted(os.listdir(anno_dir))
    actor_tracks = {}
    for i in range(0, len(anno_files), 2):
        curr_anno_file = os.path.join(anno_dir, anno_files[i])
        with gzip.open(curr_anno_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        world2ego = data["bounding_boxes"][0]["world2ego"]
        curr_timestamp = int((i / 20) * 1e9)

        for bbox in data["bounding_boxes"]:
            if bbox["class"] == "ego_vehicle" or "world_cord" not in bbox \
                or bbox["num_points"] < MIN_LIDAR_POINTS_THRESHOLD:
                continue

            vertices = np.array(bbox["world_cord"])
            vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
            point_ego = np.dot(world2ego, vertices_homo.T)
            if not PROJ_LIDAR2EGO:
                point_lidar = np.dot(ego2lidar, point_ego).T
                center, yaw = compute_obb_location_and_yaw_ordered(point_lidar)
            else:
                center, yaw = compute_obb_location_and_yaw_ordered(point_ego.T)

            yaw = float(np.round(yaw, 3))
            center = np.round(center, 3)[:3].tolist()
            extend = np.array(bbox["extent"]) * 2
            extend = np.round(extend, 3).tolist()
            track_id = int(bbox["id"])
            obj_cls, obj_attr = name2cls_attr(bbox["type_id"])
            
            if track_id not in actor_tracks:
                actor_tracks[track_id] = {
                    "track_id" : track_id,
                    "object_type" : obj_cls,
                    "dimensions": [extend],
                    "timestamps": [curr_timestamp],
                    "positions": [center],
                    "orientations": [yaw],
                    "attributes": obj_attr
                }
                
            else:
                actor_tracks[track_id]["timestamps"].append(curr_timestamp)
                actor_tracks[track_id]["positions"].append(center)
                actor_tracks[track_id]["orientations"].append(yaw)
    return actor_tracks


def preprocess_lidar(work_dir, save_dir):
    lidar_files = sorted(os.listdir(f"{work_dir}/lidar"))
    save_path = f"{save_dir}/middle_lidar"
    os.makedirs(save_path, exist_ok=True)
    for i in range(0, len(lidar_files), 2):
        curr_timestamp = int((i / 20) * 1e9)
        curr_lidar_file = os.path.join(work_dir, "lidar", lidar_files[i])
        las = laspy.read(curr_lidar_file)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        if not PROJ_LIDAR2EGO:
            points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
            new_point = (ego2lidar @ points_homo.T).T
        else:
            new_point = points
        np.savez_compressed(f"{save_path}/{curr_timestamp}.npz",
                            x=new_point[:, 0],
                            y=new_point[:, 1],
                            z=new_point[:, 2],
                            intensity=np.ones_like(new_point[:, 0]),
                            timestamp=np.ones_like(new_point[:, 0]) * curr_timestamp)

def create_symlink(work_dir, save_path):
    for src, tar in FOLDER_NANEM_MAP.items():
        try:
            src_ln = os.path.join(work_dir, "camera", src)
            assert os.path.isdir(src_ln)
            os.symlink(src_ln, os.path.join(save_path, tar))
        except FileExistsError:
            print(f"{os.path.join(save_path, tar)} already exists")
            
    
def process_scene(scene, base_work_dir, base_save_path):
    """
    Worker function: This performs all processing for a *single* scene.
    (This function is identical to the previous version)
    """
    try:
        curr_dir = os.path.join(base_work_dir, scene)
        anno_dir = os.path.join(curr_dir, "anno")
        curr_save_path = os.path.join(base_save_path, scene)
        
        # 1. Create directory
        if not os.path.exists(curr_save_path):
            os.makedirs(curr_save_path, exist_ok=True)
        elif os.path.exists(os.path.join(curr_save_path, f"{scene}.json")):
            logging.warning(f"[SUCCESS] Scene already processed: {scene}", exc_info=True)
            return f"[SUCCESS] Scene already processed: {scene}"

        # 2. Run processing steps
        get_calib_json(anno_dir, curr_save_path)
        create_symlink(curr_dir, curr_save_path)
        create_timesync_file(curr_dir, curr_save_path)
        preprocess_lidar(curr_dir, curr_save_path)
        
        # 3. Process annotations
        new_anno = {"id": scene, "timestampe" : 0.0}
        actor_tracks = preprocess_anno(anno_dir)
        new_anno["tracks"] = list(actor_tracks.values())
        
        # 4. Save the new annotation file
        output_json_path = os.path.join(curr_save_path, f"{scene}.json")
        with open(output_json_path, "w") as f:
            json.dump(new_anno, f, indent=4)
            
        # Return a success message
        return f"[SUCCESS] Processed scene: {scene}"
        
    except Exception as e:
        # Catch and report errors from within the process
        logging.error(f"Error processing scene {scene}: {e}", exc_info=True)
        return f"[ERROR] Failed to process scene: {scene} - {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default="/home/xujun/DrivIng_carla",
                        help='Root directory of the dataset')
    parser.add_argument("--split_name", type=str, default="v1_train_val.json",)
    parser.add_argument("--num_workers", type=int, default=32,)
    args = parser.parse_args()
    
    work_dir = f"{args.dataset_root}/data/drivIng_long"
    save_path = f"{args.dataset_root}/converted"
    split_path = f"{args.dataset_root}/splits/{args.split_name}"
    logging.basicConfig(filename="tmp/logging.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    train = split_data["train"]
    train = [train.split('/')[-1] for train in train]
    val = split_data["val"]
    val = [val.split('/')[-1] for val in val]
    
    process_func_with_args = partial(process_scene, 
                                     base_work_dir=work_dir, 
                                     base_save_path=save_path)
    n_cores = args.num_workers
    train_results = pqdm(train, process_func_with_args, n_jobs=n_cores)
    val_results = pqdm(val, process_func_with_args, n_jobs=n_cores)
    
    success_count = 0
    error_count = 0
    for res in train_results + val_results:
        if res.startswith("[SUCCESS]"):
            success_count += 1
        else:
            error_count += 1
            print(res) # Print error messages
            
    print(f"\nSummary: {success_count} scenes processed successfully.")
    print(f"Summary: {error_count} scenes failed.")
        
        
