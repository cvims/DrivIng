import os
import rospy
from pathlib import Path
import csv
import numpy as np
import cv2
import json


def read_file_paths(path, file_type):
    files = sorted(Path(path).glob(f'*{file_type}'))
    return files


def read_lidar_file_paths(path, file_type='.npz'):
    return read_file_paths(path, file_type)


def read_adma_file_paths(path, file_type='.json'):
    return read_file_paths(path, file_type)


def read_camera_file_paths(path, file_type='.jpg'):
    return read_file_paths(path, file_type)


def utc_string_to_rostime(utc_string):
    secs, nsecs = utc_string.split('.')
    return rospy.Time(int(secs), int(nsecs))


def rostime_to_float(rostime):
    return rostime.secs + rostime.nsecs / 1_000_000_000


def read_timesync_file(csv_path):
    """
    Loads a timesync_info.csv file and reconstructs the synchronization dictionary.

    Returns:
        dict: keys are sensor names (and 'timestamp_ms'), values are lists of entries:
              - for timestamp_ms: list of ints
              - for others: list of (timestamp, suffix, Path) tuples or None if missing
    """
    sync_data = {}
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # First row is header: ["", "1", "2", "3", ...]
    keys = []
    for row in rows[1:]:  # skip header
        key = row[0]
        keys.append(key)
        sync_data[key] = []

    num_columns = len(rows[0]) - 1

    # For each column, gather values for each key
    for col_idx in range(1, num_columns + 1):
        for row_idx, key in enumerate(keys, start=1):
            val = rows[row_idx][col_idx]

            if key == 'timestamp_nanoseconds':
                # Timestamp column, convert to int
                if val == '':
                    sync_data[key].append(None)
                else:
                    sync_data[key].append(int(val))

            else:
                if val == '' or val is None:
                    sync_data[key].append(('', ''))
                else:
                    # val like "1732632693924.jpg" or "1732632674700.npz"
                    stem = Path(val).stem
                    suffix = Path(val).suffix
                    try:
                        timestamp = int(stem)
                    except ValueError:
                        # fallback if not int, try float and convert to int ms
                        timestamp = int(float(stem))
                    sync_data[key].append((timestamp, suffix))

    return sync_data


def create_time_sync_file(sync_data, output_path, filename="timesync_info.csv", print_success=True):
    """
    Writes synchronized data to a CSV file.
    sync_data {key: [(timestamp, file_ending), ...]}
        key is the name of the sensor.
        str(timestamp) + file_ending is the file name of a file of a specific sensor (key).

    Format:
    - First column: the data source name (keys of sync_data)
    - Remaining columns: filenames (or empty string if None) for each timestamped sync frame
    """
    output_file = os.path.join(output_path, filename)
    os.makedirs(output_path, exist_ok=True)

    keys = list(sync_data.keys())
    num_columns = len(sync_data[keys[0]])  # number of sync frames

    # First row = frame indices
    header = [""] + [str(i) for i in range(1, num_columns)]
    rows = [header]

    # Build transposed rows: each row is [key, val_1, val_2, ..., val_n]
    for key in keys:
        row = [key]
        for i in range(num_columns):
            val = sync_data[key][i]
            if val is None:
                row.append("")
            else:
                # convert to new format [ts+file_ext]
                if isinstance(val, int):
                    # timestamp_nanoseconds
                    val = val
                else:
                    val = str(val[0]) + val[1]
                row.append(val)  # handles timestamp float
        rows.append(row)

    # Write CSV
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    if print_success:
        print(f"Time sync file written to: {output_file}.")


class Undistorter:
    def __init__(self, mapx, mapy):
        self.mapx = mapx
        self.mapy = mapy

    def __call__(self, frame):
        if frame is not None and frame.size > 0:
            return cv2.remap(frame, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR)
        else:
            return None


def get_fisheye_undistort_function(K, D, K_undistortion, image_size):
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_undistortion, image_size, cv2.CV_32FC1)

    return Undistorter(mapx, mapy)


def get_fisheye_intrinsic_matrix(camera_params, alpha=0):
    K = np.asarray(camera_params['IntrinsicMatrix']).reshape(3, 3).T
    D = np.asarray(camera_params['DistortionCoefficients'])

    image_size = tuple(np.flip(camera_params['ImageSize']))  # imageSize --> w x h (e.g. 640 x 512)

    K_undistortion = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, np.eye(3), balance=alpha)

    return K, D, K_undistortion, image_size, get_fisheye_undistort_function(K, D, K_undistortion, image_size)


def get_pinhole_undisort_function(K, D, K_undistortion, image_size):
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, D, None, K_undistortion, image_size, cv2.CV_32FC1)

    return Undistorter(mapx, mapy)


def get_pinhole_intrinsic_matrix(camera_params, alpha=0):
    K = np.asarray(camera_params['IntrinsicMatrix']).reshape(3, 3).T

    radialDistortion = np.asarray(camera_params['RadialDistortion'])
    tangentialDistortion = np.asarray(camera_params['TangentialDistortion'])
    D = np.insert(radialDistortion, 2, tangentialDistortion)

    image_size = tuple(np.flip(camera_params['ImageSize']))  # imageSize --> w x h (eg. 640 x 512)

    K_undistortion = cv2.getOptimalNewCameraMatrix(K, D, image_size, alpha=alpha)[0]

    return K, D, K_undistortion, image_size, get_pinhole_undisort_function(K, D, K_undistortion, image_size)


def get_camera_intrinsics(camera_params):
    camera_model = 'fisheye' if 'DistortionCoefficients' in camera_params.keys() else 'pinhole'

    return get_pinhole_intrinsic_matrix(camera_params) if camera_model == 'pinhole' else get_fisheye_intrinsic_matrix(camera_params)


def get_camera_extrinsics(camera_params):
    return np.asarray(camera_params['cTv'])


def get_lidar_extrinsics(lidar_params):
    return np.asarray(lidar_params['vTl'])


def get_adma_extrinsics(adma_params):
    return np.asarray(adma_params['vTa'])


def get_vehicle_dimension(dim_params):
    return np.array(dim_params)  # [l, w, h]


def load_calibration_file(file_path):
    with open(file_path, 'r') as f:
        calibration = json.load(f)

    data = {
        'cameras': {},
        'lidars': {},
        'vehicle': {},
        'adma': {}
    }

    for key, value in calibration.items():
        if '_camera' in key:
            intrinsics = value.get('intrinsics', {})
            extrinsics = value.get('extrinsics', {})

            if not intrinsics or not extrinsics:
                continue

            # Intrinsic matrix with correction
            K, D, K_undistortion, image_size, undistort_function = get_camera_intrinsics(intrinsics)

            # Extrinsic matrix (camera to vehicle)
            cTv = get_camera_extrinsics(extrinsics)

            data['cameras'][key] = {
                'K': K,
                'D': D,
                'K_undistortion': K_undistortion,
                'image_size': image_size,
                'undistort_function': undistort_function,
                'cTv': cTv
            }

        elif '_lidar' in key:
            extrinsics = value.get('extrinsics', {})
            if not extrinsics:
                continue
            vTl = get_lidar_extrinsics(extrinsics)
            data['lidars'][key] = {
                'vTl': vTl
            }

        elif key == 'state':
            dimension = value.get('dimension', None)
            if dimension is not None:
                data['vehicle']['dimension'] = get_vehicle_dimension(dimension)
        
        elif key == 'adma':
            extrinsics = value.get('extrinsics', {})
            if not extrinsics:
                continue
            vTa = get_adma_extrinsics(extrinsics)
            data['adma'] = {
                'vTa': vTa
            }

    return data
