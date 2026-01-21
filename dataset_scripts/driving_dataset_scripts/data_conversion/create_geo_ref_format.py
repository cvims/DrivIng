# This script is for generating boxes based on the geo-referenced data of our real-world driving.
# It uses a annotation file and the corresponding timesync_info.csv file for calculation.
import os
import argparse
import json
import tqdm
from pyproj import Transformer
import numpy as np
from scipy.spatial.transform import Rotation as R

from driving_dataset_scripts.utils.common import read_timesync_file, load_calibration_file
from driving_dataset_scripts.utils.annotations import load_annotation_data, convert_labels_from_tracks_wise_to_timestamp_wise
from driving_dataset_scripts.utils.adma import load_adma_file_data


def create_data_structure():
    return dict(
        timestamps=list(),
        ego_positions=list(),
        ego_dimensions=None,
        data=list()
    )


def transformer_lon_lat_mslheight_to_x_y_z():
    return Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)  # lon, lat, h -> X,Y,Z


def transformer_xyz_to_lon_lat_mslheight():
    return Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)  # X,Y,Z -> lon,lat,h


def calculate_lon_lat_ego(adma_lon_lat_heightmsl, adma_pitch_roll_yaw, vTa):
    lon, lat, height_msl = adma_lon_lat_heightmsl
    roll, pitch, yaw = adma_pitch_roll_yaw
    t_adma2veh = vTa[:3, 3]  # x,y,z (forward, right, up)

    geo_to_ecef = transformer_lon_lat_mslheight_to_x_y_z()
    X_adma, Y_adma, Z_adma = geo_to_ecef.transform(lon, lat, height_msl)
    ecef_adma = np.array([X_adma, Y_adma, Z_adma])

    # ADMA convention: roll (x-forward), pitch (y-right), yaw (z-up, heading from north, clockwise)
    R_body2ecef = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

    # Rotate translation into ECEF
    t_ecef = R_body2ecef.dot(t_adma2veh)

    # Apply offset
    ecef_vehicle = ecef_adma + t_ecef

    # Convert back to lon, lat, h
    ecef_to_geo = transformer_xyz_to_lon_lat_mslheight()
    lon_v, lat_v, h_v = ecef_to_geo.transform(ecef_vehicle[0], ecef_vehicle[1], ecef_vehicle[2])

    return lon_v, lat_v, h_v


def enu_axes_from_llh(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    east  = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat)*np.cos(lon),
                      -np.sin(lat)*np.sin(lon),
                       np.cos(lat)])
    up    = np.array([ np.cos(lat)*np.cos(lon),
                       np.cos(lat)*np.sin(lon),
                       np.sin(lat)])
    return np.vstack([east, north, up]).T  # 3x3 matrix ENU->ECEF


def calculate_lon_lat_object_position(object_position, object_yaw_rad, vTl, vehicle_lon_lat_heightmsl, vehicle_roll_pitch_yaw):
    """
    Project object from LiDAR frame into global geodetic coordinates.

    Parameters
    ----------
    object_position : np.ndarray
        (x, y, z) in LiDAR frame [m].
    object_yaw_rad : float
        Object yaw (heading) in LiDAR frame [rad].
    vTl : np.ndarray
        4x4 homogeneous transform from LiDAR -> vehicle.
    vehicle_lon_lat_heightmsl : tuple
        Vehicle (lon [deg], lat [deg], height_msl [m]).
    vehicle_pitch_roll_yaw : tuple
        Vehicle (roll [deg], pitch [deg], yaw [deg]) from ADMA.

    Returns
    -------
    (lon, lat, h, yaw_global) : tuple
        Object geodetic position and heading [deg].
    """
    lon, lat, height_msl = vehicle_lon_lat_heightmsl
    roll, pitch, yaw = vehicle_roll_pitch_yaw

    # LiDAR -> Vehicle
    obj_lidar_h = np.append(object_position, 1.0)  # homogeneous
    obj_vehicle = vTl @ obj_lidar_h  # position in vehicle frame (x,y,z)

    # Vehicle geodetic -> ECEF
    geo_to_ecef = transformer_lon_lat_mslheight_to_x_y_z()
    X_v, Y_v, Z_v = geo_to_ecef.transform(lon, lat, height_msl)
    ecef_vehicle = np.array([X_v, Y_v, Z_v])

    R_enu2ecef = enu_axes_from_llh(lon, lat)

    # Vehicle orientation (roll, pitch, yaw in deg)
    R_body2enu = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

    R_body2ecef = R_enu2ecef @ R_body2enu

    # Object offset in vehicle frame (already X-forward, Y-left, Z-up)
    t_obj_ecef = obj_vehicle[:3].copy()
    t_obj_ecef[1] = obj_vehicle[0]
    t_obj_ecef[0] = -obj_vehicle[1]
    t_obj_ecef = R_body2ecef @ t_obj_ecef

    # Object GPS
    ecef_object = ecef_vehicle + t_obj_ecef

    # Back to geodetic
    ecef_to_geo = transformer_xyz_to_lon_lat_mslheight()
    lon_o, lat_o, h_o = ecef_to_geo.transform(*ecef_object)

    # Object yaw (LiDAR -> Vehicle -> ECEF)
    dir_local = np.array([np.cos(object_yaw_rad), np.sin(object_yaw_rad), 0.0])
    dir_vehicle = vTl[:3, :3] @ dir_local
    dir_ecef = R_body2ecef @ dir_vehicle

    # Rotation from ECEF → ENU (transpose of R_enu2ecef)
    R_ecef2enu = R_enu2ecef.T
    dir_enu = R_ecef2enu @ dir_ecef

    # Azimuth in ENU: atan2(East, North)
    azimuth = np.degrees(np.arctan2(dir_enu[0], dir_enu[1]))  # 0°=north, 90°=east

    return lon_o, lat_o, h_o, float(azimuth)


def convert_annotation_data_to_geo_ref_format(data_path, timesync_content, annotation_data, calibration_data):
    data_structure = create_data_structure()

    vTa = calibration_data['adma']['vTa']
    vTl = calibration_data['lidars']['middle_lidar']['vTl']

    # add ego dimensions static
    data_structure['ego_dimensions'] = calibration_data['vehicle']['dimension'].tolist()  # l,w,h

    for i, timestamp_nanoseconds in tqdm.tqdm(enumerate(timesync_content['timestamp_nanoseconds']), total=len(timesync_content['timestamp_nanoseconds']), desc='Creating gps object references format.'):
        # Read ADMA data
        adma_filename = str(timesync_content['adma'][i][0]) + timesync_content['adma'][i][1]
        adma_file_path = os.path.join(data_path, 'vehicle_state', adma_filename)
        adma_data = load_adma_file_data(adma_file_path)

        # ADMA data
        ins_lon = adma_data['long_abs']  # longitude
        ins_lat = adma_data['lat_abs']  # latitude
        ins_height_msl = adma_data['height_msl']  # above sea level
        ins_roll = adma_data['roll']  # deg
        ins_pitch = adma_data['pitch']  # deg
        ins_yaw = adma_data['yaw']  # deg

        # calculate vehicle geometric center
        ego_position = calculate_lon_lat_ego((ins_lon, ins_lat, ins_height_msl), (ins_roll, ins_pitch, ins_yaw), vTa)
        data_structure['ego_positions'].append([ego_position[0], ego_position[1], ego_position[2]])

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert annotation data into geo-referenced format."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        help="Root directory containing the rosbags_converted sequences."
    )

    parser.add_argument(
        "--sequence-name",
        type=str,
        help="Name of the sequence to process."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default='',
        required=False,
        help="Name of output directory."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir

    args.sequence_name = 'day'
    args.data_root = '/nvme_data2/s3_ibatrack_cvims_v1/rosbags_processed/final_dataset_structure/'

    out_file_name = f'{args.sequence_name}_geo_ref_format.json'
    out_file_path = os.path.join(out_dir, out_file_name)

    data_path = os.path.join(args.data_root, args.sequence_name)
    timesync_file_path = os.path.join(data_path, 'timesync_info.csv')
    annotation_file_path = os.path.join(data_path, 'annotations.json')
    calibration_file_path = os.path.join(data_path, 'calibration.json')

    # Load data
    timesync_content = read_timesync_file(timesync_file_path)

    annotation_data = load_annotation_data(annotation_file_path)
    annotation_data = convert_labels_from_tracks_wise_to_timestamp_wise(annotation_data)

    calibration_data = load_calibration_file(calibration_file_path)

    # Convert
    converted_data = convert_annotation_data_to_geo_ref_format(
        data_path=data_path,
        timesync_content=timesync_content,
        annotation_data=annotation_data,
        calibration_data=calibration_data,
    )

    # Write output
    os.makedirs(out_dir, exist_ok=True)

    with open(out_file_path, 'w') as f:
        json.dump(converted_data, f, indent=4)

    print('Saved file to:', out_file_path)


if __name__ == "__main__":
    main()
