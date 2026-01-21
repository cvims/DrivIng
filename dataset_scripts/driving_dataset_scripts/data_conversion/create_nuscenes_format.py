import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
import argparse

from collections import defaultdict
import json
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, cpu_count
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import uuid
from pyproj import Geod
from driving_dataset_scripts.utils.common import load_calibration_file, read_timesync_file
from driving_dataset_scripts.utils.lidar import read_extended_point_cloud_format_npz
from driving_dataset_scripts.utils.annotations import convert_labels_from_tracks_wise_to_timestamp_wise
from driving_dataset_scripts.utils.adma import load_adma_file_data


def get_attribute_options_by_id(_id):
    if _id == 1:
        return ['adult', 'child']
    elif _id == 2:
        return ['emergency', 'regular', 'public_transport']
    elif _id == 3:
        return ['car_trailer', 'truck_trailer', 'cyclist_trailer']
    elif _id == 4:
        return ['s', 'w', 'si']  # standing, walking, sitting
    elif _id == 5:
        return ['s', 'r', 'p']  # standing, riding, pushing
    elif _id == 6:
        return ['bendy', 'rigid']
    else:
        raise ValueError(f'ID: {_id} not found for attribute options.')


def get_nuscenes_object_types(track):
    driving_object_type = track['object_type']
    driving_object_attribute = track['attributes']

    nuscenes_obj_class = None

    if driving_object_type == 'Car' or driving_object_type == 'Van':
        nuscenes_obj_class = 'vehicle.car'

    elif driving_object_type == 'Bus':
        attr_id = 6
        attr_option = driving_object_attribute[str(attr_id)]
        if attr_option not in get_attribute_options_by_id(attr_id):
            raise ValueError(f'Entry {attr_option} for attribute ID {attr_id} not found.')
        if attr_option == 'bendy':
            nuscenes_obj_class = 'vehicle.bus.bendy'
        else:
            nuscenes_obj_class = 'vehicle.bus.rigid'

    elif driving_object_type == 'Truck':
        nuscenes_obj_class = 'vehicle.truck'

    elif driving_object_type == 'OtherVehicle':
        nuscenes_obj_class = 'vehicle.truck'

    elif driving_object_type == 'Trailer':
        nuscenes_obj_class = 'vehicle.trailer'

    elif driving_object_type == 'Cyclist' or driving_object_type == 'E-Scooter':
        nuscenes_obj_class = 'vehicle.bicycle'

    elif driving_object_type == 'Motorcycle':
        nuscenes_obj_class = 'vehicle.motorcycle'

    elif driving_object_type == 'Pedestrian':
        attr_id = 1
        attr_option_1 = driving_object_attribute[str(attr_id)]
        if attr_option_1 not in get_attribute_options_by_id(attr_id):
            raise ValueError(f'Entry {attr_option_1} for attribute ID {attr_id} not found.')

        if attr_option_1 == 'adult':
            nuscenes_obj_class = 'human.pedestrian.adult'
        else:
            nuscenes_obj_class = 'human.pedestrian.child'

    elif driving_object_type == 'OtherPedestrian':
        nuscenes_obj_class = 'human.pedestrian.adult'

    elif driving_object_type == 'Animal':
        nuscenes_obj_class = 'animal'

    elif driving_object_type == 'Other':
        # TODO: double-check if only traffic objects are in here
        nuscenes_obj_class = 'movable_object.barrier'
    
    if nuscenes_obj_class is None:
        raise ValueError(f'Object type {nuscenes_obj_class} unknown.')
    
    return nuscenes_obj_class


def get_nuscenes_attribute(track):
    driving_object_type = track['object_type']
    driving_object_attribute = track['attributes']

    nuscenes_attr = None

    if driving_object_type == 'Car' or driving_object_type == 'Van':
        attr_id = 2
        attr_option = driving_object_attribute[str(attr_id)]
        if not attr_option in get_attribute_options_by_id(attr_id):
            raise ValueError(f'Entry {attr_option} for attribute ID {attr_id} not found.')

        nuscenes_attr = f'vehicle.{attr_option}'

    elif driving_object_type == 'Bus':       
        attr_id = 2
        attr_option = driving_object_attribute[str(attr_id)]
        if attr_option not in get_attribute_options_by_id(attr_id):
            raise ValueError(f'Entry {attr_option} for attribute ID {attr_id} not found.')

        nuscenes_attr = f'vehicle.{attr_option}'

    elif driving_object_type == 'Truck':
        attr_id = 2
        attr_option = driving_object_attribute[str(attr_id)]
        if attr_option not in get_attribute_options_by_id(attr_id):
            raise ValueError(f'Entry {attr_option} for attribute ID {attr_id} not found.')

        nuscenes_attr = f'vehicle.{attr_option}'

    elif driving_object_type == 'OtherVehicle':
        nuscenes_attr = 'vehicle.regular'

    elif driving_object_type == 'Trailer':
        attr_id = 3
        attr_option = driving_object_attribute[str(attr_id)]

        if attr_option not in get_attribute_options_by_id(attr_id):
            raise ValueError(f'Entry {attr_option} for attribute ID {attr_id} not found.')

        nuscenes_attr = f'vehicle.{attr_option}'

    elif driving_object_type == 'Cyclist' or driving_object_type == 'E-Scooter':
        attr_id = 5
        attr_option = driving_object_attribute[str(attr_id)]
        if attr_option not in get_attribute_options_by_id(attr_id):
            raise ValueError(f'Entry {attr_option} for attribute ID {attr_id} not found.')

        if attr_option == 's':
            nuscenes_attr = 'cycle.without_rider'
        else:
            nuscenes_attr = 'cycle.with_rider'


    elif driving_object_type == 'Motorcycle':
        nuscenes_attr = 'cycle.with_rider'

    elif driving_object_type == 'Pedestrian':
        attr_id = 4
        attr_option_4 = driving_object_attribute[str(attr_id)]
        if attr_option_4 not in get_attribute_options_by_id(attr_id):
            raise ValueError(f'Entry {attr_option_4} for attribute ID {attr_id} not found.')

        if attr_option_4 == 's':
            nuscenes_attr ='pedestrian.standing'
        elif attr_option_4 == 'w':
            nuscenes_attr ='pedestrian.walking'
        else:
            nuscenes_attr = 'pedestrian.sitting'

    elif driving_object_type == 'OtherPedestrian':
        nuscenes_attr = 'pedestrian.walking'

    elif driving_object_type == 'Animal':
        nuscenes_attr = None

    elif driving_object_type == 'Other':
        nuscenes_attr = None
    
    return nuscenes_attr


def latlon_to_enu(lat, lon, alt, origin_lat, origin_lon, origin_alt):
    geod = Geod(ellps="WGS84")
    az12, az21, dist = geod.inv(origin_lon, origin_lat, lon, lat)  # returns forward/back azimuth, distance
    # Compute ENU coordinates
    east = dist * np.sin(np.deg2rad(az12))
    north = dist * np.cos(np.deg2rad(az12))
    up = alt - origin_alt
    return np.array([east, north, up])


def calculate_gTv(adma_data, vTa, origin_lat=None, origin_lon=None, origin_alt=None):
    """
    Calculate gTv (vehicle -> global) from ADMA/INS and vTa.
    
    adma_data: dict with keys ['ins_long_abs','ins_lat_abs','ins_height_msl','ins_roll','ins_pitch','ins_yaw']
    vTa: 4x4 numpy array (vehicle -> ADMA)
    origin_lat/lon/alt: reference for ENU conversion (first frame)
    """
    # --- 1. Extract ADMA values ---
    lon = adma_data['ins_long_abs']
    lat = adma_data['ins_lat_abs']
    alt = adma_data['ins_height_msl']

    if origin_lat is None or origin_lon is None or origin_alt is None:
        origin_lat, origin_lon, origin_alt = lat, lon, alt

    # --- 2. Convert lat/lon to local ENU using geodesic ---
    geod = Geod(ellps="WGS84")
    az12, az21, dist = geod.inv(origin_lon, origin_lat, lon, lat)  # lon, lat order
    east = dist * np.sin(np.deg2rad(az12))
    north = dist * np.cos(np.deg2rad(az12))
    up = alt - origin_alt
    t_global = np.array([east, north, up])

    # --- 3. Build rotation matrix from roll, pitch, yaw ---
    roll = np.deg2rad(adma_data['ins_roll'])
    pitch = np.deg2rad(adma_data['ins_pitch'])
    yaw = np.deg2rad(adma_data['ins_yaw'])
    R_global = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

    # --- 4. Build gT_a (ADMA -> global) ---
    gTa = np.eye(4)
    gTa[:3, :3] = R_global
    gTa[:3, 3] = t_global

    # --- 5. Compute gT_v (vehicle -> global) ---
    gTv = gTa @ np.linalg.inv(vTa)

    return gTv


def process_one_sequence(arg):
    sensor_tokens, category_tokens, attribute_tokens, visibility_token_dict, root_folder, target_sample_folder, fused_lidar_topic, sequence, store_raw_data, split = arg
    scene_token = create_token()
    log_token = create_token()
    
    seq_obj = ExtractNuscenesSequence(
        sensor_tokens, category_tokens, attribute_tokens, scene_token, visibility_token_dict,
        root_folder, target_sample_folder,
        sequence, store_raw_data=store_raw_data, fused_lidar_topic=fused_lidar_topic, split=split
    )
    seq_sample_tokens = sorted(list(seq_obj.sample_tokens.items()), key=lambda x: x[0])
    
    return {
        "scene_token": scene_token,
        "log_token": log_token,
        "sequence": sequence,
        "seq_sample_tokens": seq_sample_tokens,
        "sample_table": seq_obj.sample_table,
        "sample_data_table": seq_obj.sample_data_table,
        "ego_pose_table": seq_obj.ego_pose_table,
        "calibrated_sensor_table": seq_obj.calibrated_sensor_table,
        "sample_annotation_table": seq_obj.sample_annotation_table,
        "instance_table": seq_obj.instance_table,
    }

class ExtractCustomNuscenes():
    def __init__(self, root_folder, sequences, target_nuscenes_folder, version='v1.0-trainval', train_seq_names=None, use_multiprocessing=False):
        self.root_folder = root_folder
        self.val_seq_names = [sequence for sequence in sequences if sequence not in train_seq_names]
        self.target_nuscenes_folder = target_nuscenes_folder 
        self.target_table_folder = os.path.join(self.target_nuscenes_folder, version)
        self.target_sample_folder = os.path.join(self.target_nuscenes_folder, 'samples')
        self.version = version
        self.train_seq_names = train_seq_names

        os.makedirs(self.target_table_folder, exist_ok=True)
        os.makedirs(self.target_sample_folder, exist_ok=True)

        self.use_multiprocessing = use_multiprocessing
        self.lidar_topic = "middle_lidar"
        self.category_table, self.category_tokens, self.attribute_table, self.attribute_tokens = self.create_category_and_attribute_information()
        self.sensor_table, self.sensor_tokens = self.create_sensor_information()
             
        self.scene_table = []
        self.log_table = []  

        self.sample_table = []
        self.sample_data_table = []
        self.ego_pose_table = []
        self.calibrated_sensor_table = []
        self.sample_annotation_table = []
        self.instance_table = []

        self.visibility_table = [self.create_visibility_entry(dummy=True)]
        self.visibility_token_dict = {vis["level"]: vis["token"] for vis in self.visibility_table}
        
        if version == 'v1.0-trainval':
            print("Process training sequences:")
            self.create_sequence_dependent_information(self.train_seq_names, 'train')
            
            print("Process validation sequences:")
            self.create_sequence_dependent_information(self.val_seq_names, 'val')

        elif version == 'v1.0-test':
            print("Process test sequences:")
            self.create_sequence_dependent_information(self.val_seq_names, 'test')
        else:
            print(f"{version} is invalid. Please utilze 'v1.0-trainval' or 'v1.0-test'")
        
        self.map_table = []
        self.map_table.append(self.create_map_entry(dummy=True))

    def create_sensor_information(self):
        sensor_table = []
        sensor_token_dict = {}

        for cam_name in ['back_left_camera', 'left_camera', 'front_left_camera', 'front_right_camera', 'right_camera', 'back_right_camera']:
            sensor_token = create_token()
            sensor_token_dict[cam_name] = sensor_token
            modality = "camera" 
            sensor_table.append(self.create_sensor_entry(sensor_token, cam_name, modality))
            sensor_path = os.path.join(self.target_sample_folder, cam_name)
            os.makedirs(sensor_path, exist_ok=True)
        
        sensor_token = create_token()
        sensor_token_dict[self.lidar_topic] = sensor_token
        sensor_table.append(self.create_sensor_entry(sensor_token, self.lidar_topic, "lidar"))
        sensor_path = os.path.join(self.target_sample_folder, self.lidar_topic)
        os.makedirs(sensor_path, exist_ok=True)
        return sensor_table, sensor_token_dict
    
    def create_category_and_attribute_information(self):
        category_table = []
        category_token_dict = {}

        attribute_table = []
        attribute_token_dict = {}

        def create_info(sequences, split_name):
            for sequence in sequences:
                with open(os.path.join(self.root_folder, split_name, sequence, 'annotations.json'), 'r') as f:
                    labels_trackwise = json.load(f)
                for track in labels_trackwise['tracks']:
                    nuscenes_obj_type = get_nuscenes_object_types(track)
                    if nuscenes_obj_type not in category_token_dict.keys():
                        category_token = create_token()
                        category_token_dict[nuscenes_obj_type] = category_token
                        category_table.append(self.create_category_entry(category_token, nuscenes_obj_type))

                labels_timestamp_wise = convert_labels_from_tracks_wise_to_timestamp_wise(labels_trackwise)
                for timestamp_nanoseconds in labels_timestamp_wise:
                    for track in labels_timestamp_wise[timestamp_nanoseconds]:
                        nuscenes_attribute = get_nuscenes_attribute(track)
                        if nuscenes_attribute is None:
                            continue

                        if nuscenes_attribute not in attribute_token_dict.keys():
                            attribute_token = create_token()
                            attribute_token_dict[nuscenes_attribute] = attribute_token
                            attribute_table.append(self.create_attribute_entry(attribute_token, nuscenes_attribute))

        create_info(self.val_seq_names, split_name='val' if 'trainval' in self.version else 'test')

        create_info(self.train_seq_names, split_name='train')

        return category_table, category_token_dict, attribute_table, attribute_token_dict

    def create_agent_information(self, agents):
        ego_agent_table = []
        ego_agent_token = {}
        for agent in agents:
            agent_token = create_token()
            ego_agent_table.append(self.create_agent_entry(agent_token, agent))
            ego_agent_token[agent] = agent_token
        
        return ego_agent_table, ego_agent_token

    def create_sequence_dependent_information(self, sequences, split):
        store_raw_data = True
        results = []
        args_list = []
        for sequence in sequences:
            args_list.append((
                self.sensor_tokens,
                self.category_tokens,
                self.attribute_tokens,
                self.visibility_token_dict,
                self.root_folder,
                self.target_sample_folder,
                self.lidar_topic,
                sequence,
                store_raw_data,
                split
            ))

        if self.use_multiprocessing:
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(process_one_sequence, args_list)
    
        else:
            for arg in args_list:
                results.append(process_one_sequence(arg))

        for result in results:
            scene_entry = self.create_scene_entry(
                result["scene_token"], result["log_token"],
                result["sequence"], len(result["seq_sample_tokens"]),
                result["seq_sample_tokens"][0][1], result["seq_sample_tokens"][-1][1]
            )
            log_entry = self.create_log_entry(result["log_token"], date=result["sequence"].split('_')[0])

            self.scene_table.append(scene_entry)
            self.log_table.append(log_entry)
            self.sample_table += result["sample_table"]
            self.sample_data_table += result["sample_data_table"]
            self.ego_pose_table += result["ego_pose_table"]
            self.calibrated_sensor_table += result["calibrated_sensor_table"]
            self.sample_annotation_table += result["sample_annotation_table"]
            self.instance_table += result["instance_table"]

        store_raw_data = False
 
    def create_visibility_entry(self, dummy=False):
        if dummy:
            return {
                "token": create_token(),
                "level": "1",
                "description": "Fully visible"
            }
        
    def create_map_entry(self, dummy=False):
        if dummy:
            return {
                "category": "",
                "token": create_token(),
                "filename": "",
                "log_tokens": [log["token"] for log in self.log_table],
            }
        
    def create_category_entry(self, category_token, category, description = None):
        return {
            "token": category_token,
            "name": category,
            "description": description,
        }

    def create_attribute_entry(self, attribute_token, attribute, description=None):
        return {
            "token": attribute_token,
            "name": attribute,
            "description": description
        }

    def create_sensor_entry(self, sensor_token, topic, modality):
       if modality == None:
            return None
       return {
            "token": sensor_token,
            "channel": topic,
            "modality": modality
        }

    def create_agent_entry(self, agent_token, agent):
        return {
            "token": agent_token,
            "name": agent
        }

    def create_scene_entry(self, scene_token, log_token, sequence_name, nbr_samples, first_sample_token, last_sample_token):
        return {
            "token": scene_token,
            "log_token": log_token,
            "nbr_samples": nbr_samples,
            "first_sample_token": first_sample_token,
            "last_sample_token": last_sample_token,
            "name": sequence_name, 
            "description": ""
        }

    def create_log_entry(self, log_token, date):
        return {
            "token": log_token,
            "logfile": "",
            "date_captured": date,
            "location": "Ingolstadt",
            "vehicle": "Audi Q8 e-tron"
        }
                
    def write_table(self, name, data):
        with open(os.path.join(self.target_table_folder, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=2)

    def write_split(self):
        scene_names = [scene['name'] for scene in self.scene_table]
        split_path = os.path.join(self.target_table_folder, "split")
        os.makedirs(split_path, exist_ok=True)
        if "test" in self.version:
            file_name = os.path.join(split_path, 'test.txt')
            with open(file_name, 'w') as f:
                for name in scene_names:
                    f.write(name + '\n')
        else:
            val_scenes = [scene_name for scene_name in scene_names if scene_name not in self.train_seq_names]
            train_scenes = self.train_seq_names 
            with open(os.path.join(split_path, 'train.txt'), 'w') as f:
                for name in train_scenes:
                    f.write(name + '\n')

            with open(os.path.join(split_path, 'val.txt'), 'w') as f:
                for name in val_scenes:
                    f.write(name + '\n')

    def token_exists(self, token, table):
        exists = any([True if entry['token'] == token else False for entry in table])
        return exists

    def prev_and_next_exist(self, table):
        for entry in tqdm(table):
            prev_token = entry['prev']
            prev_exists = self.token_exists(prev_token, table) if prev_token != '' else True
            next_token = entry['next']
            next_exists = self.token_exists(next_token, table) if next_token != '' else True
            if not (prev_exists and next_exists):
                return False
        return True

    def first_and_last_ann_of_instance_exist(self):
        for entry in tqdm(self.instance_table):
            first_ann_token = entry["first_annotation_token"]
            first_exists = self.token_exists(first_ann_token, self.sample_annotation_table)
            last_ann_token = entry["last_annotation_token"]
            last_exists = self.token_exists(last_ann_token, self.sample_annotation_table)
            if not (first_exists and last_exists):
                return False
        return True
    
    def test_tables(self):
        ann_references_good = self.prev_and_next_exist(self.sample_annotation_table)
        sample_references_good = self.prev_and_next_exist(self.sample_table)
        sample_data_references_good = self.prev_and_next_exist(self.sample_data_table)

        instances_good = self.first_and_last_ann_of_instance_exist()

        all_references_good = ann_references_good and sample_data_references_good and sample_references_good and instances_good

        return all_references_good

    def write_tables(self):
        tables = {
            "scene": self.scene_table,
            "sample": self.sample_table,
            "sample_data": self.sample_data_table,
            "sample_annotation": self.sample_annotation_table,
            "instance": self.instance_table,
            "category": self.category_table,
            "attribute": self.attribute_table,
            "visibility": self.visibility_table,
            "sensor": self.sensor_table,
            "calibrated_sensor": self.calibrated_sensor_table,
            "ego_pose": self.ego_pose_table,
            "log": self.log_table,
            "map": self.map_table,
            # "ego_agents": self.ego_agent_table, 
        }
        for name, data in tables.items():
            self.write_table(name, data)

class ExtractNuscenesSequence():
    def __init__(self, sensor_tokens, category_tokens, attribute_tokens, scene_token, visibility_tokens, root_folder, target_sample_folder, sequence, store_raw_data=False, fused_lidar_topic=None, split=None):
        self.fused_lidar_topic = fused_lidar_topic
        self.sensor_tokens = sensor_tokens
        self.category_tokens = category_tokens
        self.attribute_tokens = attribute_tokens
        self.root_folder = root_folder
        self.target_sample_folder = target_sample_folder
        self.store_raw_data = store_raw_data
        self.sequence = sequence
        self.scene_token = scene_token
        self.visibility_tokens = visibility_tokens
        self.split = split
        self.load_dataset_information()
        self.initialize_tokens_of_sequence()

        self.sample_table = []
        self.sample_data_table = []
        self.ego_pose_table = []
        self.calibrated_sensor_table = []
        self.sample_annotation_table = []
        self.instance_table = []
        self.fill_raw_data_independent_information()
        self.fill_raw_data_dependent_information()

    def initialize_tokens_of_sequence(self):
        self.instance_tokens = {}
        self.sample_annotation_tokens = defaultdict(dict)
        self.calib_tokens = defaultdict(dict)
        self.sample_tokens = {}
        self.sample_data_tokens = defaultdict(dict)
        self.pose_tokens = {}
        self.track_id2category = {}
        for timestamp_nanoseconds in self.time_sync_df["timestamp_nanoseconds"]:
            if timestamp_nanoseconds in self.labels_tswise:
                labels_of_ts = self.labels_tswise[timestamp_nanoseconds]
                for track in labels_of_ts:
                    track_id = track['track_id']
                    ann_token = create_token()
                    self.sample_annotation_tokens[track_id][timestamp_nanoseconds] = ann_token
                    if track_id not in self.track_id2category.keys():
                        nuscenes_obj_type = get_nuscenes_object_types(track)
                        self.track_id2category[track_id] = nuscenes_obj_type

        for track_id in self.track_id2category.keys():
            self.instance_tokens[track_id] = create_token()
         
        for timestamp_nanoseconds in self.time_sync_df["timestamp_nanoseconds"]:
            sample_token = create_token()
            self.sample_tokens[timestamp_nanoseconds] = sample_token
            pose_token = create_token()
            self.pose_tokens[timestamp_nanoseconds] = pose_token

            for sensor in self.sensor_tokens.keys():
                sample_data_token = create_token()
                calib_token = create_token()
                self.sample_data_tokens[sensor][timestamp_nanoseconds] = sample_data_token
                self.calib_tokens[sensor][timestamp_nanoseconds] = calib_token 

    def fill_raw_data_independent_information(self):
        for track_id, category in self.track_id2category.items():
            self.instance_table.append(self.create_instance_entry(track_id, category))

        for i, timestamp_nanoseconds in enumerate(self.time_sync_df['timestamp_nanoseconds']):
            prev_ts_nanoseconds = self.time_sync_df['timestamp_nanoseconds'][i-1] if i > 0 else None
            next_ts_nanoseconds = self.time_sync_df['timestamp_nanoseconds'][i+1] if i < len(self.time_sync_df['timestamp_nanoseconds']) - 1 else None

            self.sample_table.append(self.create_sample_entry(timestamp_nanoseconds, prev_ts_nanoseconds, next_ts_nanoseconds))

    def load_dataset_information(self):
        calibration_file = os.path.join(self.root_folder, self.split, self.sequence, 'calibration.json')
        time_sync_info_file = os.path.join(self.root_folder, self.split, self.sequence, 'timesync_info.csv')

        self.calib_data = load_calibration_file(calibration_file)

        self.time_sync_df = read_timesync_file(time_sync_info_file)
        for topic in self.time_sync_df:
            self.time_sync_df[topic] = self.time_sync_df[topic]

        with open(os.path.join(self.root_folder, self.split, self.sequence, 'annotations.json'), 'r') as f:
            labels_trackwise = json.load(f)

        self.labels_tswise = convert_labels_from_tracks_wise_to_timestamp_wise(labels_trackwise, skip_tracks=[]) # no tracks skipped 

    def fill_raw_data_dependent_information(self):
        sensor_types = list([k for k in self.time_sync_df.keys() if k != 'timestamp_nanoseconds'])

        # origin lon, lat, alt from first adma file
        adma_filename = str(self.time_sync_df['vehicle_state'][0][0]) + self.time_sync_df['vehicle_state'][0][1]
        adma_file_path = os.path.join(self.root_folder, self.split, self.sequence, 'vehicle_state', adma_filename)
        adma_data = load_adma_file_data(adma_file_path)

        origin_lon = adma_data['long_abs']
        origin_lat = adma_data['lat_abs']
        origin_alt = adma_data['height_msl']

        # adma to vehicle (vTa)
        vTa = self.calib_data['adma']['vTa']

        for i, timestamp_nanoseconds in tqdm(
            enumerate(self.time_sync_df['timestamp_nanoseconds']),
                desc=f"Processing all tracks of sequence {self.sequence}.",
                total=len(self.time_sync_df['timestamp_nanoseconds'])):
        
            # Calculate ego to global (gTv)
            adma_filename = str(self.time_sync_df['vehicle_state'][i][0]) + self.time_sync_df['vehicle_state'][i][1]
            adma_file_path = os.path.join(self.root_folder, self.split, self.sequence, 'vehicle_state', adma_filename)
            adma_data = load_adma_file_data(adma_file_path)

            # gTv = calculate_gTv(adma_data, vTa, origin_lat, origin_lon, origin_alt)
            gTv = np.eye(4)
            
            self.ego_pose_table.append(self.create_pose_entry(timestamp_nanoseconds, gTv))

            prev_ts_nanoseconds=self.time_sync_df['timestamp_nanoseconds'][i-1] if i > 0 else None
            next_ts_nanoseconds=self.time_sync_df['timestamp_nanoseconds'][i+1] if i < len(self.time_sync_df['timestamp_nanoseconds']) - 1 else None

            for sensor_type in sensor_types:
                sensor_timestamp_info = self.time_sync_df[sensor_type][i]
                if sensor_type.endswith('camera'):
                    egoTs = np.linalg.inv(self.calib_data['cameras'][sensor_type]['cTv'])
                    sensor_root_path = os.path.join(self.target_sample_folder, sensor_type)
                    iTc = self.calib_data['cameras'][sensor_type]['K_undistortion']
                    if self.sequence:
                        filename = os.path.join(sensor_root_path, '_'.join([self.sequence, f'{timestamp_nanoseconds}.jpg']))
                    else:
                        filename = os.path.join(sensor_root_path, f'{timestamp_nanoseconds}.jpg')
                    fileformat = "jpg"
                    if sensor_timestamp_info is None or sensor_timestamp_info[0] is None:
                        image = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)
                    else:
                        image_distorted = cv2.imread(os.path.join(self.root_folder, self.split, self.sequence, sensor_type, str(sensor_timestamp_info[0]) + str(sensor_timestamp_info[1])))
                        image = self.calib_data['cameras'][sensor_type]['undistort_function'](image_distorted)
                    if image is None:
                        image = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)
                    hw = image.shape[:2]
                    if self.store_raw_data:
                        cv2.imwrite(filename, image)
                elif sensor_type.endswith('lidar'):
                    egoTs = self.calib_data['lidars'][sensor_type]['vTl']
                    sensor_root_path = os.path.join(self.target_sample_folder, sensor_type)
                    if self.sequence:
                        filename = os.path.join(sensor_root_path, '_'.join([self.sequence, f'{timestamp_nanoseconds}.pcd.bin']))
                    else:
                        filename = os.path.join(sensor_root_path, f'{timestamp_nanoseconds}.pcd.bin')
                    fileformat = "pcd"
                    hw = None
                    iTc = None
                    org_lidar_file_path = os.path.join(self.root_folder, self.split, self.sequence, sensor_type, str(sensor_timestamp_info[0]) + str(sensor_timestamp_info[1]))
                    pcd = read_extended_point_cloud_format_npz(org_lidar_file_path)  # x, y, z, intensity, timestamp (nuscenes uses: xyz, intensity, ring index)

                    # pcd = (egoTs[:3, :3] @ pcd[:,:3].T).T

                    pcd = pcd.astype(np.float32)
                    pcd[:, 4] = 0.0  # replace timestamp with fake ring index (0)

                    pcd.tofile(filename)
                else:
                    continue

                self.calibrated_sensor_table.append(self.create_calib_entry(timestamp_nanoseconds, sensor_type, egoTs, iTc))

                self.sample_data_table.append(
                    self.create_sample_data_entry(
                        timestamp_nanoseconds, sensor_type, '/'.join(filename.split('/')[-3:]), fileformat, hw,
                        prev_ts_nanoseconds=prev_ts_nanoseconds,
                        next_ts_nanoseconds=next_ts_nanoseconds)
                )

            self.create_ann_entries_for_current_ts(
                timestamp_nanoseconds, pcd,
                prev_ts_nanoseconds=prev_ts_nanoseconds,
                next_ts_nanoseconds=next_ts_nanoseconds,
                vTl=self.calib_data['lidars']['middle_lidar']['vTl'])
    
    def create_ann_entries_for_current_ts(self, timestamp_nanoseconds, pcd_lidar, prev_ts_nanoseconds, next_ts_nanoseconds, vTl):
        if timestamp_nanoseconds not in self.labels_tswise:
            return
        obj_list = self.labels_tswise[timestamp_nanoseconds]

        for obj in obj_list:
            # position + orientation in LiDAR frame
            obj_pos_lidar = np.array(obj['position'])
            dims = np.array(obj['dimension'])  # l, w, h
            dims_wlh = np.array([dims[1], dims[0], dims[2]])  # convert to w,l,h

            # Rotation matrix (LiDAR frame)
            R_obj_lidar = Rotation.from_euler('z', obj['orientation']).as_matrix()

            # Transform to vehicle frame
            obj_pos_vehicle = (vTl[:3, :3] @ obj_pos_lidar) + vTl[:3, 3]
            R_obj_vehicle = vTl[:3, :3] @ R_obj_lidar

            # Transform points to vehicle frame
            pcd_vehicle = (vTl[:3, :3] @ pcd_lidar[:,:3].T).T + vTl[:3,3]

            # Relative points to box center
            pcd_rel = pcd_vehicle - obj_pos_vehicle

            # Rotate into box frame
            pcd_aligned = (R_obj_vehicle.T @ pcd_rel.T).T

            # Shift z to geometric center if GT is bottom-face centered
            # pcd_aligned[:,2] += dims[2]/2

            # Filter inside box
            min_corner = -dims / 2
            max_corner = dims / 2
            mask = np.all((pcd_aligned >= min_corner) & (pcd_aligned <= max_corner), axis=1)

            # Quaternion (NuScenes convention)
            obj_rot_quat = rotation_matrix2nuscenes_quaternion(R_obj_vehicle)

            attribute_token = get_nuscenes_attribute(obj)
            if attribute_token is None:
                attribute_tokens = []
            else:
                attribute_tokens = [self.attribute_tokens[attribute_token]]

            # Save annotation
            self.sample_annotation_table.append(
                self.create_sample_annotations_entry(
                    obj_pos_vehicle.tolist(),
                    dims_wlh.tolist(),
                    obj_rot_quat,
                    obj['track_id'],
                    timestamp_nanoseconds,
                    num_lidar_pts=int(mask.sum()),
                    attribute_tokens=attribute_tokens,
                    prev_ts_nanoseconds=prev_ts_nanoseconds,
                    next_ts_nanoseconds=next_ts_nanoseconds
                )
            )


    # helper: draw one box on BEV image
    def draw_box_on_bev(self, bev_img, box_center, dims, yaw, scale=1, bev_shape=(1000, 2000), color=(255,0,0)):
        """
        Draw a 3D box in BEV (top-down) view.

        box_center: (x,y) in meters, vehicle frame
        dims: (l,w) in meters
        yaw: rotation around z-axis in radians
        scale: meters to pixels
        bev_shape: H,W of output image
        """
        l, w = dims[:2]  # length, width
        cx, cy = box_center[:2]

        # box corners relative to center
        corners = np.array([
            [ l/2,  w/2],
            [ l/2, -w/2],
            [-l/2, -w/2],
            [-l/2,  w/2]
        ])
        # rotate corners
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
        corners_rot = (corners @ R.T) + np.array([cx, cy])

        # convert to BEV pixel coords
        x_img = ((corners_rot[:, 0] + 100) / 0.1).astype(int)
        y_img = ((corners_rot[:, 1] + 100) / 0.1).astype(int)

        # assemble polygon
        pts = np.stack([x_img, y_img], axis=-1)

        # draw polygon
        color = (0, 255, 0)  # green box for now
        cv2.polylines(bev_img, [pts], isClosed=True, color=color, thickness=2)


    def timestamp_nanoseconds_to_microseconds(self, timestamp_nanoseconds):
        return int(timestamp_nanoseconds / 1e3)

    def create_pose_entry(self, timestamp_nanoseconds, gTego):
        return {
            "token": self.pose_tokens[timestamp_nanoseconds],
            "timestamp": self.timestamp_nanoseconds_to_microseconds(timestamp_nanoseconds),
            "translation": gTego[:-1, -1].tolist(),
            "rotation": rotation_matrix2nuscenes_quaternion(gTego[:-1, :-1]).tolist(),
        }

    def create_calib_entry(self, timestamp_nanoseconds, key, egoTs, iTc):
        return {
            "token": self.calib_tokens[key][timestamp_nanoseconds],
            "sensor_token": self.sensor_tokens[key],
            "translation": egoTs[:-1, -1].tolist(),
            "rotation": rotation_matrix2nuscenes_quaternion(egoTs[:-1, :-1]).tolist(),
            "camera_intrinsic": iTc.tolist() if iTc is not None else []
        }      
       
    def create_instance_entry(self, track_id, category):
        sorted_anns = sorted(list(self.sample_annotation_tokens[track_id].items()), key= lambda x: x[0])
        return {
            "token": self.instance_tokens[track_id],
            "category_token": self.category_tokens[category],
            "nbr_annotations": len(self.sample_annotation_tokens[track_id]),
            "first_annotation_token": sorted_anns[0][1],
            "last_annotation_token": sorted_anns[-1][1], # ego dependent because ann is sample_token dependent
        }
    
    def create_sample_annotations_entry(self, obj_xyz, obj_wlh, obj_rot_quaternion, track_id, timestamp_nanoseconds, num_lidar_pts, attribute_tokens, prev_ts_nanoseconds, next_ts_nanoseconds):
        return {
            "token": self.sample_annotation_tokens[track_id][timestamp_nanoseconds],
            "sample_token": self.sample_tokens[timestamp_nanoseconds],
            "instance_token": self.instance_tokens[track_id],
            "visibility_token": self.visibility_tokens["1"], 
            "attribute_tokens": attribute_tokens,
            "translation": obj_xyz,
            "size": obj_wlh, # nuscenes uses width, length, height
            "rotation": obj_rot_quaternion.tolist(), # Quaternion [w, x, y, z]
            "prev": self.sample_annotation_tokens[track_id].get(prev_ts_nanoseconds, ''),
            "next": self.sample_annotation_tokens[track_id].get(next_ts_nanoseconds, ''),  # nuscenes: Track is a new instance if it temporarily disappears => next is null if it does not appear in the next Frame
            "num_lidar_pts": num_lidar_pts,
            "num_radar_pts": 0,
        }

    def create_sample_entry(self, timestamp_nanoseconds, prev_ts_nanoseconds, next_ts_nanoseconds):
        return {
            "token": self.sample_tokens[timestamp_nanoseconds],
            "timestamp": self.timestamp_nanoseconds_to_microseconds(timestamp_nanoseconds),
            "prev": self.sample_tokens.get(prev_ts_nanoseconds, '') if prev_ts_nanoseconds is not None else '',
            "next":  self.sample_tokens.get(next_ts_nanoseconds, '') if next_ts_nanoseconds is not None else '',
            "scene_token": self.scene_token
        }
    
    def create_sample_data_entry(self, timestamp_nanoseconds, key, filename, fileformat, hw, prev_ts_nanoseconds, next_ts_nanoseconds):
        return {
            "token": self.sample_data_tokens[key][timestamp_nanoseconds],
            "sample_token": self.sample_tokens[timestamp_nanoseconds],
            "ego_pose_token": self.pose_tokens[timestamp_nanoseconds],
            "calibrated_sensor_token": self.calib_tokens[key][timestamp_nanoseconds],
            "filename": filename,
            "fileformat": fileformat,
            "timestamp": self.timestamp_nanoseconds_to_microseconds(timestamp_nanoseconds),
            "is_key_frame": True, # uncertain about meaning
            "height": hw[0] if hw is not None else None,
            "width": hw[1] if hw is not None else None,
            "prev": self.sample_data_tokens[key].get(prev_ts_nanoseconds, '') if prev_ts_nanoseconds is not None else '',
            "next": self.sample_data_tokens[key].get(next_ts_nanoseconds, '') if next_ts_nanoseconds is not None else '',
        }
    
def create_token():
    return str(uuid.uuid4())

def rotation_matrix2nuscenes_quaternion(R):
    R = Rotation.from_matrix(R).as_quat(scalar_first=True)
    return R


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract custom nuScenes tables from labeled sequences."
    )

    parser.add_argument(
        "--split-path",
        type=str,
        help="Root folder containing the split sequences (train/val/test)."
    )

    parser.add_argument(
        "--sequence-name",
        type=str,
        default="day",
        help="Name of the sequence to process."
    )

    parser.add_argument(
        "--target-nuscenes-folder",
        type=str,
        required=False,
        help="Root folder where nuScenes-formatted tables will be written."
    )

    parser.add_argument(
        "--use-multiprocessing",
        action="store_true",
        help="Whether to use multiprocessing when generating tables."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    split_path = args.split_path
    full_split_path = os.path.join(split_path, args.sequence_name)

    if not os.path.exists(full_split_path):
        raise FileNotFoundError(f"Full split path does not exist: {full_split_path}")

    train_dir = os.path.join(full_split_path, 'train')
    val_dir = os.path.join(full_split_path, 'val')
    test_dir = os.path.join(full_split_path, 'test')

    train_sequences = [seq for seq in os.listdir(train_dir)]
    val_sequences = [seq for seq in os.listdir(val_dir)]
    test_sequences = [seq for seq in os.listdir(test_dir)]

    target_nuscenes_folder = args.target_nuscenes_folder
    if not args.target_nuscenes_folder:
        target_nuscenes_folder = os.path.join(split_path, '..', 'nuScenes_DrivIng')

    target_folder = os.path.join(target_nuscenes_folder, args.sequence_name)
    os.makedirs(target_folder, exist_ok=True)

    # Process train+val sequences
    trainval_sequences = list(set(train_sequences + val_sequences))
    CustomNuscenesObject = ExtractCustomNuscenes(
        full_split_path,
        trainval_sequences,
        target_folder,
        version='v1.0-trainval',
        train_seq_names=train_sequences,
        use_multiprocessing=args.use_multiprocessing
    )
    CustomNuscenesObject.write_tables()

    # Process test sequences
    CustomNuscenesObject = ExtractCustomNuscenes(
        full_split_path,
        test_sequences,
        target_folder,
        version='v1.0-test',
        train_seq_names=[],
        use_multiprocessing=args.use_multiprocessing
    )
    CustomNuscenesObject.write_tables()


if __name__ == "__main__":
    freeze_support()
    main()
