import os
import json
import math
from typing import List
import pandas as pd
import numpy as np
import carla
import re
import sys
from pyproj import Proj
from pathlib import Path
import csv
import copy
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

WORKING_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(WORKING_DIR, "dataset_scripts"))
sys.path.append(os.path.join(WORKING_DIR, "dataset_scripts" , "driving_dataset_scripts", "data_conversion"))
sys.path.append(os.path.join(WORKING_DIR, "dataset_scripts" , "driving_dataset_scripts", "utils"))
from common import read_timesync_file, load_calibration_file
from create_geo_ref_format import calculate_lon_lat_ego, calculate_lon_lat_object_position

PROJ_STR = "+proj=tmerc +lat_0=0 +lon_0=9 +k=0.9996 +x_0=-179849.9999 +y_0=-5403979.9998 +datum=WGS84 +units=m +no_defs"
PROJ_FUNC = Proj(PROJ_STR)

CARLA_VEHICLE_TYPES = [
    "vehicle.mini.cooper_s",
    "vehicle.audi.etron",
    "vehicle.chevrolet.impala",
    "vehicle.dodge.charger_2020",
    "vehicle.dodge.charger_police",
    "vehicle.dodge.charger_police_2020",
    "vehicle.lincoln.mkz_2017",
    "vehicle.lincoln.mkz_2020",
    "vehicle.mini.cooper_s_2021",
    "vehicle.mercedes.coupe_2020",
    "vehicle.ford.mustang",
    "vehicle.nissan.patrol_2021",
    "vehicle.audi.tt",
    "vehicle.audi.etron",
    "vehicle.ford.crown",
    "vehicle.ford.mustang",
    "vehicle.tesla.model3",
    "vehicle.jeep.wrangler_rubicon",
    "vehicle.mercedes.coupe",
    "vehicle.micro.microlino",
    "vehicle.mini.cooper_s",
    "vehicle.nissan.micra",
    "vehicle.nissan.patrol",
    "vehicle.seat.leon",
    "vehicle.toyota.prius",
    "vehicle.tesla.model3",
    "vehicle.bmw.grandtourer",
    "vehicle.audi.a2",
    "vehicle.citroen.c3",
]

CARLA_MOTOCYCLE_TYPES = [
    "vehicle.harley-davidson.low_rider",
    "vehicle.kawasaki.ninja",
    "vehicle.vespa.zx125",
    "vehicle.yamaha.yzf",
]

CARLA_BICYCLE_TYPES = [
    "vehicle.bh.crossbike",
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
]

def load_annotation_data(labels_path):
    if labels_path is None:
        return None

    with open(os.path.join(labels_path), 'r') as f:
        labels_trackwise = json.load(f)
    label_timewise = convert_labels_from_tracks_wise_to_timestamp_wise(labels_trackwise)
    
    return label_timewise


def convert_labels_from_tracks_wise_to_timestamp_wise(labels, skip_tracks=[]):
    if labels is None:
        return None

    labels_new = {}

    for track in labels['tracks']:
        if track['track_id'] in skip_tracks:
            continue
        for timestamp_index, timestamp in enumerate(track['timestamps']):
            if timestamp not in labels_new.keys():
                labels_new[timestamp] = []

            this_dict = {
                'track_id': track['track_id'],
                'object_type': track['object_type'],
                'position': track['positions'][timestamp_index],
                'orientation': track['orientations'][timestamp_index],
                'dimension': track['dimensions'][0] if len(track['dimensions']) == 1 else track['dimensions'][timestamp_index],
                'attributes': dict(),
            }

            for k, v in track['attributes'].items():
                if isinstance(v, list):
                    this_dict['attributes'][k] = v[timestamp_index]
                else:
                    this_dict['attributes'][k] = v

            labels_new[timestamp].append(this_dict)

    return labels_new

def get_ego_yaw_in_enu(roll, pitch, yaw_adma):
    """
    Converts ADMA (NED-based) orientation to ENU Yaw.
    """
    # 1. Represent the ADMA orientation
    # ADMA uses 'xyz' order (Roll, Pitch, Yaw)
    r_adma = R.from_euler('xyz', [roll, pitch, yaw_adma], degrees=True)

    # 2. Extract the Forward Vector of the car [X, Y, Z]
    # In the car's frame, Forward is [1, 0, 0]
    # We rotate this vector to see where it points in the global frame
    forward_vec = r_adma.apply([1, 0, 0])

    # 3. Calculate the angle in the ENU plane
    # forward_vec[0] is East (X), forward_vec[1] is North (Y)
    # atan2(y, x) gives the standard mathematical angle (0=East, 90=North)
    yaw_enu_rad = np.arctan2(forward_vec[1], forward_vec[0])
    
    return np.degrees(yaw_enu_rad)

def convert_annotation_data_to_geo_ref_format(data_path, timesync_content, annotation_data, calibration_data):
    data_structure = dict(
        timestamps=list(),
        ego_positions=list(),
        ego_dimensions=None,
        data=list()
    )

    vTa = calibration_data['adma']['vTa']
    vTl = calibration_data['lidars']['middle_lidar']['vTl']

    # add ego dimensions static
    data_structure['ego_dimensions'] = calibration_data['vehicle']['dimension'].tolist()  # l,w,h

    for i, timestamp_nanoseconds in tqdm(enumerate(timesync_content['timestamp_nanoseconds']), total=len(timesync_content['timestamp_nanoseconds']), desc='Creating gps object references format.'):
        # Read ADMA data
        adma_data = None
        adma_filename = str(timesync_content['vehicle_state'][i][0]) + timesync_content['vehicle_state'][i][1]
        adma_file_path = os.path.join(data_path, 'vehicle_state', adma_filename)
        with open(adma_file_path, 'r') as file:
            adma_data = json.load(file)

        # ADMA data
        ins_lon = adma_data['long_abs']  # longitude
        ins_lat = adma_data['lat_abs']  # latitude
        ins_height_msl = adma_data['height_msl']  # above sea level
        ins_roll = adma_data['roll']  # deg
        ins_pitch = adma_data['pitch']  # deg
        ins_yaw = adma_data['yaw']  # deg

        # calculate vehicle geometric center
        ego_position = calculate_lon_lat_ego((ins_lon, ins_lat, ins_height_msl), (ins_roll, ins_pitch, ins_yaw), vTa)
        ego_yaw = get_ego_yaw_in_enu(ins_roll, ins_pitch, ins_yaw)
        data_structure['ego_positions'].append([ego_position[0], ego_position[1], ego_position[2], ins_yaw])
        
        # add timestamp to data structure
        data_structure['timestamps'].append(timestamp_nanoseconds)

        if timestamp_nanoseconds in annotation_data:
            # Read data
            data_list = []
            for track in annotation_data[timestamp_nanoseconds]:
                lon_o, lat_o, h_o, azimuth = calculate_lon_lat_object_position(
                    track['position'], track['orientation'], vTl,
                    ego_position, (ins_roll, ins_pitch, ins_yaw)
                )
                position = [lon_o, lat_o, h_o]

                data_dict = dict(
                    track_id=track['track_id'],
                    object_type=track['object_type'],
                    dimension=track['dimension'],
                    position=position,
                    azimuth=azimuth,
                )
                data_list.append(data_dict)
            # add to data
            data_structure['data'].append(data_list)
        else:
            data_structure['data'].append([dict()])

    return data_structure

def gps_to_carla_location(lat, lon):
    easting, northing = PROJ_FUNC(lon, lat)
    return easting, -northing

def location_to_gps(x, y):
    easting, northing = PROJ_FUNC(x, -y, inverse=True)
    return easting, -northing


def relative_to_world_transform(ego_transform: carla.Transform, relative_position: List[float], relative_yaw: float) -> carla.Transform:
    """
    Convert ego-relative position to world coordinates correctly.
    """
    rel_location = carla.Location(x=relative_position[1], y=-relative_position[0], z=relative_position[2])
    world_location = ego_transform.transform(rel_location)
    final_yaw = -(ego_transform.rotation.yaw + relative_yaw)

    final_rotation = carla.Rotation(yaw=final_yaw)

    return carla.Transform(world_location, final_rotation)

def convert_depth(data):
    """
    Computes the normalized depth from a CARLA depth map.
    """
    data = data.astype(np.float16)

    normalized = np.dot(data, [65536.0, 256.0, 1.0])
    normalized /= (256 * 256 * 256 - 1)
    return normalized * 1000

def project_to_aligned_lane_waypoint(map_obj, ref_location, is_vehicle=True):
    """
    Project position to nearest drivable lane waypoint with yaw angle alignment.
    
    Args:
        map_obj: CARLA map object
        x, y: Position coordinates (from GPS conversion)
        z: Height coordinate (optional, defaults to 0.0)
        yaw: Yaw angle in degrees (heading direction)
        tolerance_meters: Maximum distance tolerance (default 5.0m)
    
    Returns:
        Waypoint object if within tolerance and aligned, otherwise None
    """
    
    # First, get the closest waypoint
    if is_vehicle:
        closest_waypoint = map_obj.get_waypoint(
            ref_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
    else:
        return ref_location # skip pedestrain class

    if closest_waypoint is None:
        return ref_location
    else:
        return closest_waypoint.transform

def normalize_angle_difference(angle1, angle2):
    """
    Calculate the smallest difference between two angles in degrees.
    Returns value between -180 and 180 degrees.
    """
    diff = angle1 - angle2
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff

def calculate_2d_distance(loc1, loc2):
    """Calculate 2D Euclidean distance between two locations."""
    dx = loc1.x - loc2.x
    dy = loc1.y - loc2.y
    return math.sqrt(dx*dx + dy*dy)

def build_annotation_ref_json(anno_file, timesync_file, calib_file, dataset_root):
    annotation_data = load_annotation_data(anno_file)
    timesync_data = read_timesync_file(timesync_file)
    calibration_file = load_calibration_file(calib_file)
    annotations_geo_ref = convert_annotation_data_to_geo_ref_format(
        data_path=dataset_root,
        timesync_content=timesync_data,
        annotation_data=annotation_data,
        calibration_data=calibration_file
    )
    
    with open(os.path.join(dataset_root, 'annotations_ref.json'), 'w') as f:
        json.dump(annotations_geo_ref, f)