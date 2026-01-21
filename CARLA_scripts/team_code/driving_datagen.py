from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.envs.sensor_interface import SensorInterface, CallBack, OpenDriveMapReader, SpeedometerReader
from leaderboard.autoagents import autonomous_agent_local as autonomous_agent
from autopilot import AutoPilot
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
import carla
import gzip
from easydict import EasyDict
import math
import h5py
import laspy
import os
import datetime
import pathlib
from utils import build_projection_matrix, convert_depth, get_relative_transform, normalize_angle, build_skeleton,  get_matrix, calculate_cube_vertices, compute_2d_distance
from utils import DIS_CAR_SAVE, DIS_WALKER_SAVE, DIS_SIGN_SAVE, DIS_LIGHT_SAVE
from birds_eye_view.transforms import vec_global_to_ref
from agents.navigation.local_planner import RoadOption
import torch

EARTH_RADIUS_EQUA = 6378137.0
SAVE_PATH = os.environ.get('SAVE_PATH', None)

def get_entry_point():
    return "DriveIngDataAgent"

class DriveIngDataAgent(AutoPilot):
    frame_rate = 10.0 
    def setup(self, path_to_conf_file, route_date_string=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_date_string, traffic_manager=None)
        self.track = autonomous_agent.Track.SENSORS
        self.save_depth = True
        self.vehicle_dim = [4.915, 1.937, 1.633]


        if SAVE_PATH is not None:
            (self.save_path / 'camera').mkdir()
            (self.save_path / 'lidar').mkdir()
            (self.save_path / 'anno').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
            
            (self.save_path / 'camera' / 'rgb_left').mkdir()
            (self.save_path / 'camera' / 'rgb_front_left').mkdir()
            (self.save_path / 'camera' / 'rgb_front_right').mkdir()
            (self.save_path / 'camera' / 'rgb_right').mkdir()
            (self.save_path / 'camera' / 'rgb_back_left').mkdir()
            (self.save_path / 'camera' / 'rgb_back_right').mkdir()
            (self.save_path / 'camera' / 'rgb_top_down').mkdir()
            
                
            if self.save_depth:
                (self.save_path / 'camera' / 'depth_left').mkdir()
                (self.save_path / 'camera' / 'depth_front_left').mkdir()
                (self.save_path / 'camera' / 'depth_front_right').mkdir()
                (self.save_path / 'camera' / 'depth_right').mkdir()
                (self.save_path / 'camera' / 'depth_back_left').mkdir()
                (self.save_path / 'camera' / 'depth_back_right').mkdir()
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])

    def tick(self, input_data):
        # control
        control = self._vehicle.get_control()

        # camera_bgr
        cam_bgr_left = input_data['CAM_LEFT'][1][:, :, :3]
        cam_bgr_front_left = input_data['CAM_FRONT_LEFT'][1][:, :, :3]
        cam_bgr_front_right = input_data['CAM_FRONT_RIGHT'][1][:, :, :3]
        cam_bgr_right = input_data['CAM_RIGHT'][1][:, :, :3]
        cam_bgr_back_left = input_data['CAM_BACK_LEFT'][1][:, :, :3]
        cam_bgr_back_right = input_data['CAM_BACK_RIGHT'][1][:, :, :3]
        cam_bgr_top_down = input_data['TOP_DOWN'][1][:, :, :3]
        
        # lidar
        lidar = input_data['LIDAR_TOP']

        def lidar_to_ego_coordinate(lidar):
            """
            Converts the LiDAR points given by the simulator into the ego agents
            coordinate system
            :param config: GlobalConfig, used to read out lidar orientation and location
            :param lidar: the LiDAR point cloud as provided in the input of run_step
            :return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
            coordinate system.
            """
            lidar_rot = [-0.28, 0.17, 89.92]
            lidar_pos = [-0.49, 0.0, 2.0565]

            yaw = np.deg2rad(lidar_rot[2])
            rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

            translation = np.array(lidar_pos)

            # The double transpose is a trick to compute all the points together.
            ego_lidar = (rotation_matrix @ lidar[1][:, :3].T).T + translation

            return ego_lidar
        
        lidar = lidar_to_ego_coordinate(lidar)
        lidar_360 = lidar
        
        bounding_boxes = self.get_bounding_boxes(lidar=lidar_360)
        sensors_anno = self.get_sensors_anno()
        
        self.last_lidar = lidar
        self.last_ego_transform = self._vehicle.get_transform()
        # gps/imu
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        # cam_bgr_depth
        if self.save_depth:
            cam_bgr_left_depth = input_data['CAM_LEFT_DEPTH'][1][:, :, :3]
            cam_bgr_front_left_depth = input_data['CAM_FRONT_LEFT_DEPTH'][1][:, :, :3]
            cam_bgr_front_right_depth = input_data['CAM_FRONT_RIGHT_DEPTH'][1][:, :, :3]
            cam_bgr_right_depth = input_data['CAM_RIGHT_DEPTH'][1][:, :, :3]
            cam_bgr_back_left_depth = input_data['CAM_BACK_LEFT_DEPTH'][1][:, :, :3]
            cam_bgr_back_right_depth = input_data['CAM_BACK_RIGHT_DEPTH'][1][:, :, :3]
            
            # cam_gray_depth, 16 bit would be ideal, but we can't afford the extra storage.
            cam_gray_left_depth = convert_depth(cam_bgr_left_depth)
            cam_gray_front_left_depth = convert_depth(cam_bgr_front_left_depth)
            cam_gray_front_right_depth = convert_depth(cam_bgr_front_right_depth)
            cam_gray_right_depth = convert_depth(cam_bgr_right_depth)
            cam_gray_back_left_depth = convert_depth(cam_bgr_back_left_depth)
            cam_gray_back_right_depth = convert_depth(cam_bgr_back_right_depth)
            
        # weather
        weather = self._weather_to_dict(self._world.get_weather())

        self.cam_bgr_mapping = {
            'CAM_LEFT': 'cam_bgr_left',
            'CAM_FRONT_LEFT': 'cam_bgr_front_left',
            'CAM_FRONT_RIGHT': 'cam_bgr_front_right',
            'CAM_RIGHT': 'cam_bgr_right',
            'CAM_BACK_LEFT': 'cam_bgr_back_left',
            'CAM_BACK_RIGHT': 'cam_bgr_back_right',
        }

        self.cam_bgr_depth_mapping = {
            'CAM_LEFT': 'cam_bgr_left_depth',
            'CAM_FRONT_LEFT': 'cam_bgr_front_left_depth',
            'CAM_FRONT_RIGHT': 'cam_bgr_front_right_depth',
            'CAM_RIGHT': 'cam_bgr_right_depth',
            'CAM_BACK_LEFT': 'cam_bgr_back_left_depth',
            'CAM_BACK_RIGHT': 'cam_bgr_back_right_depth',

        }

        self.cam_gray_depth_mapping = {
            'CAM_LEFT': 'cam_gray_left_depth',
            'CAM_FRONT_LEFT': 'cam_gray_front_left_depth',
            'CAM_FRONT_RIGHT': 'cam_gray_front_right_depth',
            'CAM_RIGHT': 'cam_gray_right_depth',
            'CAM_BACK_LEFT': 'cam_gray_back_left_depth',
            'CAM_BACK_RIGHT': 'cam_gray_back_right_depth',
        }

        results = {
                # cam_bgr
                'cam_bgr_left': cam_bgr_left,
                'cam_bgr_front_left': cam_bgr_front_left,
                'cam_bgr_front_right': cam_bgr_front_right,
                'cam_bgr_right': cam_bgr_right,
                'cam_bgr_back_left': cam_bgr_back_left,
                'cam_bgr_back_right': cam_bgr_back_right,
                'cam_bgr_top_down': cam_bgr_top_down,
                
                # lidar
                'lidar' : lidar_360,
                # other
                'gps': gps,
                'speed': speed,
                'compass': compass,
                'weather': weather,
                "acceleration":acceleration,
                "angular_velocity":angular_velocity,
                'bounding_boxes': bounding_boxes,
                'sensors_anno': sensors_anno,
                'throttle': control.throttle,
                'steer': control.steer,
                'brake': control.brake,
                'reverse': control.reverse,
                'town': CarlaDataProvider.get_map(),
                }
        
        if self.save_depth:
            # save the original bgr depth, please remember to post-process the depth
            depth_results = {
                'cam_bgr_left_depth': cam_bgr_left_depth,
                'cam_bgr_front_left_depth' : cam_bgr_front_left_depth,
                'cam_bgr_front_right_depth': cam_bgr_front_right_depth,
                'cam_bgr_right_depth': cam_bgr_right_depth,
                'cam_bgr_back_left_depth': cam_bgr_back_left_depth,
                'cam_bgr_back_right_depth': cam_bgr_back_right_depth,
                
                'cam_gray_left_depth': cam_gray_left_depth,
                'cam_gray_front_left_depth': cam_gray_front_left_depth,
                'cam_gray_front_right_depth': cam_gray_front_right_depth,
                'cam_gray_right_depth': cam_gray_right_depth,
                'cam_gray_back_left_depth': cam_gray_back_left_depth,
                'cam_gray_back_right_depth': cam_gray_back_right_depth}
            results.update(depth_results)
        
        return results
    
    def sensors(self):
        sensors = [
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.54,
                    'y': -0.26,
                    'z': self.vehicle_dim[2]/2.0+1.07,
                    'roll': 90-89.98,
                    'pitch': -0.63,
                    'yaw': -72.23-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 52.12,
                    # 'fov': 36.24,
                    'id': 'CAM_BACK_LEFT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.5,
                    'y': -0.7,
                    'z': self.vehicle_dim[2]/2.0+0.9,
                    'roll': 90-89.93,
                    'pitch': -0.18,
                    'yaw': -1.7-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 87.14,
                    # 'fov': 62.14,
                    'id': 'CAM_LEFT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.24,
                    'y': -0.03,
                    'z': self.vehicle_dim[2]/2.0+1.01,
                    'roll': 90-88.5,
                    'pitch': -0.18,
                    'yaw': 69.11-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 53.13,
                    # 'fov': 36.83,
                    'id': 'CAM_FRONT_LEFT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.43,
                    'y': -0.02,
                    'z': self.vehicle_dim[2]/2.0+1.1,
                    'roll': 90-89.8,
                    'pitch': -0.25,
                    'yaw': 108.96-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 52.21,
                    # 'fov': 36.02,
                    'id': 'CAM_FRONT_RIGHT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.62,
                    'y': 0.64,
                    'z': self.vehicle_dim[2]/2.0+0.87,
                    'roll': 90-90.42,
                    'pitch': -0.17,
                    'yaw': 179.8-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 86.93,
                    # 'fov': 61.75,
                    'id': 'CAM_RIGHT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.29,
                    'y': -0.22,
                    'z': self.vehicle_dim[2]/2.0+1.01,
                    'roll': 90-89.38,
                    'pitch': 0.68,
                    'yaw': -110.43-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 49.91,
                    # 'fov': 34.19,
                    'id': 'CAM_BACK_RIGHT'
                    },
                # lidar
                {   'type': 'sensor.lidar.ray_cast',
                    'x': -0.49, 'y': 0, 'z': self.vehicle_dim[2]/2.0+1.24,
                    'roll': -0.28, 'pitch': 0.17, 'yaw': 89.92,
                    'range': 250,
                    'rotation_frequency': 20,
                    'channels': 128,
                    'points_per_second': 2500000,
                    'upper_fov': 15,
                    'lower_fov': -25,
                    'rotation_offset': 0.0,
                    'dropoff_general_rate': 0.0,
                    'dropoff_intensity_limit': 0.0,
                    'dropoff_zero_intensity': 0.0,
                    'id': 'LIDAR_TOP'
                    },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -0.3, 'y': -0.03, 'z': self.vehicle_dim[2]/2.0-0.29,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                    },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -0.3, 'y': -0.03, 'z': self.vehicle_dim[2]/2.0-0.29,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                    },
                ### Debug sensor, not used by the model
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'TOP_DOWN'
                    },
                {
                    "type": "sensor.other.imu",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "sensor_tick": 0.05,
                    "id": "imu"
                }, 
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
            ]
        
        if self.save_depth:
            depth_cameras = [
                {
                    'type': 'sensor.camera.depth',
                    'x': -0.54,
                    'y': -0.26,
                    'z': self.vehicle_dim[2]/2.0+1.07,
                    'roll': 90-89.98,
                    'pitch': -0.63,
                    'yaw': -72.23,
                    'width': 1920,
                    'height': 1080,
                    'fov': 52.12,
                    # 'fov': 36.24,
                    'id': 'CAM_BACK_LEFT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -0.5,
                    'y': -0.7,
                    'z': self.vehicle_dim[2]/2.0+0.9,
                    'roll': 90-89.93,
                    'pitch': -0.18,
                    'yaw': -1.7,
                    'width': 1920,
                    'height': 1080,
                    'fov': 87.14,
                    # 'fov': 62.14,
                    'id': 'CAM_LEFT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -0.24,
                    'y': -0.03,
                    'z': self.vehicle_dim[2]/2.0+1.01,
                    'roll': 90-88.5,
                    'pitch': -0.18,
                    'yaw': 69.11,
                    'width': 1920,
                    'height': 1080,
                    'fov': 53.13,
                    # 'fov': 36.83,
                    'id': 'CAM_FRONT_LEFT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -0.43,
                    'y': -0.02,
                    'z': self.vehicle_dim[2]/2.0+1.1,
                    'roll': 90-89.8,
                    'pitch': -0.25,
                    'yaw': 108.96,
                    'width': 1920,
                    'height': 1080,
                    'fov': 52.21,
                    # 'fov': 36.02,
                    'id': 'CAM_FRONT_RIGHT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -0.62,
                    'y': 0.64,
                    'z': self.vehicle_dim[2]/2.0+0.87,
                    'roll': 90-90.42,
                    'pitch': -0.17,
                    'yaw': 179.8,
                    'width': 1920,
                    'height': 1080,
                    'fov': 86.93,
                    # 'fov': 61.75,
                    'id': 'CAM_RIGHT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.29,
                    'y': -0.22,
                    'z': self.vehicle_dim[2]/2.0+1.01,
                    'roll': 90-89.38,
                    'pitch': 0.68,
                    'yaw': -110.43,
                    'width': 1920,
                    'height': 1080,
                    'fov': 49.91,
                    # 'fov': 34.19, 
                    'id': 'CAM_BACK_RIGHT_DEPTH'
                    }]
            sensors.extend(depth_cameras)
            
        
        self.sensors_mapping = {}
        for sensor in sensors:
            self.sensors_mapping[sensor['id']] = sensor
        return sensors

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale
        return gps
    
    def get_target_gps(self, gps, compass):
		# target gps
        def gps_to_location(gps):
            # gps content: numpy array: [lat, lon, alt]
            lat, lon, z = gps
            scale = math.cos(self.lat_ref * math.pi / 180.0)
            my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
            mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
            y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
            x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
            z = float(z)
            location = carla.Location(x=x, y=y, z=z)
            return location
            pass
        global_plan_gps = self._global_plan[:]
        next_gps, _ = global_plan_gps[min(self.navigation_idx+1, len(global_plan_gps)-1)]
        next_gps = np.array([next_gps['lat'], next_gps['lon'], next_gps['z']])
        next_vec_in_global = gps_to_location(next_gps) - gps_to_location(gps)
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass)-90.0)
        loc_in_ev = vec_global_to_ref(next_vec_in_global, ref_rot_in_global)
        
        if np.sqrt(loc_in_ev.x**2+loc_in_ev.y**2) < 12.0 and loc_in_ev.x < 0.0:
            self.navigation_idx += 1
        
        self.navigation_idx = min(self.navigation_idx, len(self._global_plan)-2)
        _, road_option_0 = global_plan_gps[max(0, self.navigation_idx)]
        gps_point, road_option_1 = global_plan_gps[self.navigation_idx+1]
        gps_point = np.array([gps_point['lat'], gps_point['lon'], gps_point['z']])
        if (road_option_0 in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]) \
                and (road_option_1 not in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]):
            road_option = road_option_1
        else:
            road_option = road_option_0

        return np.array(gps_point, dtype=np.float32), np.array([road_option.value], dtype=np.int8)

    def save_sensor(self, tick_data):
        frame = self.step
        # CARLA images are already in opencv's BGR format.
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_left' / (f'{frame:05}.jpg')), tick_data['cam_bgr_left'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_front_left' / (f'{frame:05}.jpg')), tick_data['cam_bgr_front_left'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_front_right' / (f'{frame:05}.jpg')), tick_data['cam_bgr_front_right'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_right' / (f'{frame:05}.jpg')), tick_data['cam_bgr_right'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_back_left' / (f'{frame:05}.jpg')), tick_data['cam_bgr_back_left'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_back_right' / (f'{frame:05}.jpg')), tick_data['cam_bgr_back_right'], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.save_path / 'camera' / 'rgb_top_down' / (f'{frame:05}.jpg')), tick_data['cam_bgr_top_down'], [cv2.IMWRITE_JPEG_QUALITY, 20])

        if self.save_depth:
            cv2.imwrite(str(self.save_path / 'camera' / 'depth_left' / (f'{frame:05}.png')), tick_data['cam_gray_left_depth'])
            cv2.imwrite(str(self.save_path / 'camera' / 'depth_front_left' / (f'{frame:05}.png')), tick_data['cam_gray_front_left_depth'])
            cv2.imwrite(str(self.save_path / 'camera' / 'depth_front_right' / (f'{frame:05}.png')), tick_data['cam_gray_front_right_depth'])
            cv2.imwrite(str(self.save_path / 'camera' / 'depth_right' / (f'{frame:05}.png')), tick_data['cam_gray_right_depth'])
            cv2.imwrite(str(self.save_path / 'camera' / 'depth_back_left' / (f'{frame:05}.png')), tick_data['cam_gray_back_left_depth'])
            cv2.imwrite(str(self.save_path / 'camera' / 'depth_back_right' / (f'{frame:05}.png')), tick_data['cam_gray_back_right_depth'])
            
        # Specialized LiDAR compression format
        header = laspy.LasHeader(point_format=0)  # LARS point format used for storing
        header.offsets = np.min(tick_data['lidar'], axis=0)
        point_precision = 0.001
        header.scales = np.array([point_precision, point_precision, point_precision])

        with laspy.open(self.save_path / 'lidar' / (f'{frame:05}.laz'), mode='w', header=header) as writer:
            point_record = laspy.ScaleAwarePointRecord.zeros(tick_data['lidar'].shape[0], header=header)
            point_record.x = tick_data['lidar'][:, 0]
            point_record.y = tick_data['lidar'][:, 1]
            point_record.z = tick_data['lidar'][:, 2]
            writer.write_points(point_record)

        anno_data = {
                'x': self._vehicle.get_transform().location.x,
                'y': self._vehicle.get_transform().location.y,
                'throttle': tick_data['throttle'],
                'steer': tick_data['steer'],
                'brake': tick_data['brake'],
                'reverse': tick_data['reverse'],
                'theta': tick_data['compass'],
                'speed': tick_data['speed'],
                'weather': tick_data['weather'],
                "acceleration":tick_data["acceleration"].tolist(),
                "angular_velocity":tick_data["angular_velocity"].tolist(),
                'bounding_boxes': tick_data['bounding_boxes'],
                'sensors': tick_data['sensors_anno'],
                # 'only_ap_brake': tick_data['only_ap_brake'],
                }
        with gzip.open(self.save_path / 'anno' / f'{frame:05}.json.gz', 'wt', encoding='utf-8') as f:
            json.dump(anno_data, f, indent=4)
            
    def _weather_to_dict(self, carla_weather):
        weather = {
            "cloudiness": carla_weather.cloudiness,
            "precipitation": carla_weather.precipitation,
            "precipitation_deposits": carla_weather.precipitation_deposits,
            "wind_intensity": carla_weather.wind_intensity,
            "sun_azimuth_angle": carla_weather.sun_azimuth_angle,
            "sun_altitude_angle": carla_weather.sun_altitude_angle,
            "fog_density": carla_weather.fog_density,
            "fog_distance": carla_weather.fog_distance,
            "wetness": carla_weather.wetness,
            "fog_falloff": carla_weather.fog_falloff,
        }

        return weather
            
    # add feature (yzj)
    def _point_inside_boundingbox(self, point, bb_center, bb_extent, multiplier=1.5):
        A = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
        B = carla.Vector2D(bb_center.x + multiplier * bb_extent.x, bb_center.y - multiplier * bb_extent.y)
        D = carla.Vector2D(bb_center.x - multiplier * bb_extent.x, bb_center.y + multiplier * bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad 
    
    def affected_by_traffic_light(self, traffic_light, center, window_size=50):
        # inx = min(window_size, len(CarlaDataProvider._ego_actor.route_plan))
        # if inx <= 1:
        #     return False
        # for trans, _ in CarlaDataProvider._ego_actor.route_plan[:inx]:
        #     if self._point_inside_boundingbox(trans.location, center, traffic_light.trigger_volume.extent):
        #         return True
        # return False
               
        return self._point_inside_boundingbox(self._vehicle.get_transform().location, center, traffic_light.trigger_volume.extent)

    def get_traffic_color(self, state):
        if state == carla.libcarla.TrafficLightState.Green:
            return 'green'
        if state == carla.libcarla.TrafficLightState.Yellow:
            return 'yellow'
        if state == carla.libcarla.TrafficLightState.Red:
            return 'red'
        if state == carla.libcarla.TrafficLightState.Unknown:
            return 'unknown'
        if state == carla.libcarla.TrafficLightState.Off:
            return 'off'
        raise Exception(f"{state} not in Green, Yellow, Red, Unknown, Off")
    
    def get_affect_sign(self, actors):
        all_actors = []
        affect_signs = []
        seen_actors = set()
        mini_sign = DIS_SIGN_SAVE + 1
        most_affect_sign = None
        # find all lights
        ego_vehicle_waypoint = CarlaDataProvider.get_map().get_waypoint(self._vehicle.get_location())
        for sign in actors:
            if sign.id in seen_actors:
                continue
            seen_actors.add(sign.id)
            flag = 0
            sign_loc = sign.get_location()
            if compute_2d_distance(sign_loc, self._vehicle.get_transform().location) > DIS_SIGN_SAVE:
                continue
            all_actors.append(sign)

            # find all affect lights 
            if hasattr(sign, 'trigger_volume'):
                sign_vol_loc = sign.trigger_volume.location
                sign_vol_loc_world = sign.get_transform().transform(sign_vol_loc)
                sign_vol_loc_world_wp = CarlaDataProvider.get_map().get_waypoint(sign_vol_loc_world)
                while not sign_vol_loc_world_wp.is_intersection:
                    if len(sign_vol_loc_world_wp.next(0.5)) > 0:
                        next_sign_vol_loc_world_wp = sign_vol_loc_world_wp.next(0.5)[0]
                    else:
                        flag = 1
                        break
                    if next_sign_vol_loc_world_wp and not next_sign_vol_loc_world_wp.is_intersection:
                        sign_vol_loc_world_wp = next_sign_vol_loc_world_wp
                    else:
                        break
                if flag:
                    continue
                if self.affected_by_traffic_light(sign, carla.Location(x=sign_vol_loc_world_wp.transform.location.x, y=sign_vol_loc_world_wp.transform.location.y, z=0)):
                    affect_signs.append(sign)
                    dis = np.abs(compute_2d_distance(ego_vehicle_waypoint.transform.location, sign.get_transform().transform(sign.trigger_volume.location)))
                    if dis < mini_sign:
                        most_affect_sign = sign
                        mini_sign = dis
            else:
                sign_vol_loc = sign.get_transform().location
                sign_vol_loc_world_wp = CarlaDataProvider.get_map().get_waypoint(sign_vol_loc)
                dis = compute_2d_distance(sign_vol_loc_world_wp.transform.location, ego_vehicle_waypoint.transform.location)
                forward_vec = self._vehicle.get_transform().get_forward_vector()
                ray = sign_vol_loc_world_wp.transform.location - self._vehicle.get_location()
                if forward_vec.dot(ray) < 0:
                    continue
                if dis < mini_sign:
                    most_affect_sign = sign
                    mini_sign = dis
        return all_actors, most_affect_sign

    def get_actor_filter_traffic_sign(self):
        actor_data = EasyDict({})
        speed_limit_sign = list(CarlaDataProvider.get_world().get_actors().filter("*traffic.speed_limit*")) # carla.libcarla.TrafficSign
        stop_sign = list(CarlaDataProvider.get_world().get_actors().filter("*traffic.stop*")) # carla.libcarla.TrafficSign
        yield_sign = list(CarlaDataProvider.get_world().get_actors().filter("*traffic.yield*")) # carla.libcarla.TrafficSign
        warning = list(CarlaDataProvider.get_world().get_actors().filter('*warning*'))
        dirtdebris = list(CarlaDataProvider.get_world().get_actors().filter('*dirtdebris*'))
        cone = list(CarlaDataProvider.get_world().get_actors().filter('*cone*'))

        # new custom traffic sign
        speedlimiter = list(CarlaDataProvider.get_world().get_actors().filter("*static.prop.speedlimiter*"))
        merge_sign = list(CarlaDataProvider.get_world().get_actors().filter("*static.prop.merge*"))
        arrow = list(CarlaDataProvider.get_world().get_actors().filter("*static.prop.arrow*"))
        bend = list(CarlaDataProvider.get_world().get_actors().filter("*static.prop.bend*"))
        barrier = list(CarlaDataProvider.get_world().get_actors().filter("*barrier*"))
        block = list(CarlaDataProvider.get_world().get_actors().filter("*block*"))
        pole = list(CarlaDataProvider.get_world().get_actors().filter("*static.prop.pole*"))

        actors = speed_limit_sign + stop_sign + yield_sign + warning + dirtdebris + cone
        actors += speedlimiter + merge_sign + arrow + bend + barrier + block
        actors += pole
        all_actors, most_affect_sign = self.get_affect_sign(actors)
        actor_data.actors = all_actors
        actor_data.most_affect_sign = most_affect_sign
        return actor_data

    def get_actor_filter_traffic_light(self):
        actor_data = EasyDict({})
        lights = CarlaDataProvider.get_world().get_actors().filter("*traffic_light*")
        all_lights = []
        affect_lights = []
        most_affect_light = None
        mini_lt = DIS_LIGHT_SAVE + 1

        # find all lights
        for lt in lights:
            flag = 0
            lt_loc = lt.get_location()
            if compute_2d_distance(lt_loc, self._vehicle.get_location()) > DIS_LIGHT_SAVE: # lidar range
                continue
            all_lights.append(lt)

            # find all affect lights 
            lt_vol_loc = lt.trigger_volume.location
            lt_vol_loc_world = lt.get_transform().transform(lt_vol_loc)
            lt_vol_loc_world_wp = CarlaDataProvider.get_map().get_waypoint(lt_vol_loc_world)
            while not lt_vol_loc_world_wp.is_intersection:
                if len(lt_vol_loc_world_wp.next(0.5)) > 0:
                    next_lt_vol_loc_world_wp = lt_vol_loc_world_wp.next(0.5)[0]
                else:
                    flag = 1
                    break 
                if next_lt_vol_loc_world_wp and not next_lt_vol_loc_world_wp.is_intersection:
                    lt_vol_loc_world_wp = next_lt_vol_loc_world_wp
                else:
                    break
            if flag:
                continue
            if self.affected_by_traffic_light(lt, carla.Location(x=lt_vol_loc_world_wp.transform.location.x, y=lt_vol_loc_world_wp.transform.location.y, z=0)):
                affect_lights.append(lt)
                # find most affect light_actor, min_dis=DIS_LIGHT_SAVE
                dis = np.abs(compute_2d_distance(lt.get_transform().transform(lt.trigger_volume.location), self._vehicle.get_location()))
                forward_vec = self._vehicle.get_transform().get_forward_vector()
                ray = lt.get_transform().transform(lt.trigger_volume.location) - self._vehicle.get_location()
                if forward_vec.dot(ray) < 0:
                    continue
                if dis < mini_lt:
                    most_affect_light = lt
                    mini_lt = dis
        
        actor_data.lights = all_lights
        actor_data.affect_lights = affect_lights
        actor_data.most_affect_light = most_affect_light

        #  get distance
        if most_affect_light is not None:
            trigger_volume = most_affect_light.trigger_volume
            box = trigger_volume.extent
            loc = trigger_volume.location
            ori = trigger_volume.rotation.get_forward_vector()
            trigger_loc = [loc.x, loc.y, loc.z]
            trigger_ori = [ori.x, ori.y, ori.z]
            trigger_box = [box.x, box.y]

            world_loc = most_affect_light.get_transform().transform(loc)
            world_loc_wp = CarlaDataProvider.get_map().get_waypoint(world_loc)
            while not world_loc_wp.is_intersection:
                next_world_loc_wp = world_loc_wp.next(0.5)[0]
                if next_world_loc_wp and not next_world_loc_wp.is_intersection:
                    world_loc_wp = next_world_loc_wp
                else:
                    break
            
            world_loc_wp = carla.Location(x=world_loc_wp.transform.location.x, y=world_loc_wp.transform.location.y, z=0)
            pos = self._vehicle.get_location()
            pos = carla.Location(x=pos.x, y=pos.y, z=0)

            # ego2lane_dis = world_loc_wp.distance(pos)
            ego2lane_dis = compute_2d_distance(world_loc_wp, pos)
            ego2light_dis = compute_2d_distance(most_affect_light.get_location(), self._vehicle.get_location())
            most_affect_light_id = most_affect_light.id
            most_affect_light_state = self.get_traffic_color(most_affect_light.state)

            # record data
            actor_data.most_affect_light = EasyDict()
            actor_data.most_affect_light.id = most_affect_light_id
            actor_data.most_affect_light.state = most_affect_light_state
            actor_data.most_affect_light.ego2lane_dis = ego2lane_dis
            actor_data.most_affect_light.ego2light_dis = ego2light_dis
            actor_data.most_affect_light.trigger_volume = EasyDict()
            actor_data.most_affect_light.trigger_volume.location = trigger_loc
            actor_data.most_affect_light.trigger_volume.orientation = trigger_ori
            actor_data.most_affect_light.trigger_volume.extent = trigger_box
        return actor_data 

        
    def get_actor_filter_vehicle(self):
        vehicles_dict = EasyDict({})

        vehicles = self._world.get_actors().filter('*vehicle*')
        vehicles_list = []
        for actor in vehicles:
            dist = compute_2d_distance(actor.get_transform().location, self._vehicle.get_transform().location)
            # Filter for the vehicles within DIS_CAR_SAVE m
            if dist < DIS_CAR_SAVE: # lidar range
                vehicles_list.append(actor)
        vehicles_dict.vehicle = vehicles_list

        others = self._world.get_actors().filter('*static.prop.mesh*')
        static_list = []
        for actor in others:
            # filter static vehicle
            mesh_path = actor.attributes['mesh_path'].split('/Game/Carla/')[1]
            if 'Car' in mesh_path or 'Truck' in mesh_path or 'Bus' in mesh_path or 'Motorcycle' in mesh_path or 'Bicycle' in mesh_path:
                dist = compute_2d_distance(actor.get_transform().location, self._vehicle.get_transform().location)
                # Filter for the vehicles within DIS_CAR_SAVE
                if dist < DIS_CAR_SAVE: # lidar range
                    static_list.append(actor)
        vehicles_dict.static = static_list
        return vehicles_dict
    

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed
    
    def get_sensors_anno(self):
        results = {}
        sensors = {}
        for value in self._world.get_actors().filter('*sensor.camera.rgb'):
            sensors[value.attributes['role_name']] = value

        for key in ['CAM_LEFT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'TOP_DOWN']:
            value = sensors[key]
            location = value.get_transform().location
            rotation = value.get_transform().rotation
            world2cam = value.get_transform().get_inverse_matrix()
            width = int(value.attributes['image_size_x'])
            height = int(value.attributes['image_size_y'])
            fov = float(value.attributes['fov'])
            K = build_projection_matrix(width, height, fov=fov)
            cam2ego = get_matrix(location=[self.sensors_mapping[key]['x'], self.sensors_mapping[key]['y'], self.sensors_mapping[key]['z']], 
                                 rotation=[self.sensors_mapping[key]['pitch'], self.sensors_mapping[key]['roll'], self.sensors_mapping[key]['yaw']])
            result ={
                'location': [location.x, location.y, location.z], 
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'intrinsic': K.tolist(),
                'world2cam': world2cam,
                'cam2ego': cam2ego.tolist(),
                'fov': fov,
                'image_size_x': width,
                'image_size_y': height,
            }
            results[key] = result
        
        # lidar
        for value in self._world.get_actors().filter('*sensor.lidar.ray_cast*'):
            sensors[value.attributes['role_name']] = value

        for key in ['LIDAR_TOP']:
            value = sensors[key]
            location = value.get_transform().location
            rotation = value.get_transform().rotation
            world2lidar = value.get_transform().get_inverse_matrix()
            lidar2ego = get_matrix(location=[self.sensors_mapping[key]['x'], self.sensors_mapping[key]['y'], self.sensors_mapping[key]['z']], 
                        rotation=[self.sensors_mapping[key]['pitch'], self.sensors_mapping[key]['roll'], self.sensors_mapping[key]['yaw']])
            result ={
                'location': [location.x, location.y, location.z], 
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'world2lidar': world2lidar,
                'lidar2ego': lidar2ego.tolist(),
            }
            results[key] = result
        return results

    def get_bounding_boxes(self, lidar=None, radar=None):
        results = []

        # ego_vehicle
        npc = self._vehicle
        npc_id = str(npc.id)
        npc_type_id = npc.type_id
        npc_base_type = npc.attributes['base_type']
        location = npc.get_transform().location
        rotation = npc.get_transform().rotation
        # 
        # verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
        # center, extent = get_center_and_extent(verts)
        # from carla official
        # bb_cords = _bounding_box_to_world(npc.bounding_box)
        # world_cord = _vehicle_to_world(bb_cords, npc)
        # from handcraft
        extent = npc.bounding_box.extent
        center = npc.get_transform().transform(npc.bounding_box.location)
        local_verts = calculate_cube_vertices(npc.bounding_box.location, npc.bounding_box.extent)
        global_verts = []
        for l_v in local_verts:
            g_v = npc.get_transform().transform(carla.Location(l_v[0], l_v[1], l_v[2]))
            global_verts.append([g_v.x, g_v.y, g_v.z])
        ###################
        ego_speed = self._get_forward_speed(transform=npc.get_transform(), velocity=npc.get_velocity())
        ego_brake = npc.get_control().brake
        ego_matrix = np.array(npc.get_transform().get_matrix())
        ego_yaw = np.deg2rad(rotation.yaw)
        road_id = CarlaDataProvider.get_map().get_waypoint(location).road_id
        lane_id = CarlaDataProvider.get_map().get_waypoint(location).lane_id
        section_id = CarlaDataProvider.get_map().get_waypoint(location).section_id
        world2ego = npc.get_transform().get_inverse_matrix()

        result = {
            'class': 'ego_vehicle',
            'id': npc_id,
            'type_id': npc_type_id,
            'base_type': npc_base_type,
            'location': [location.x, location.y, location.z],
            'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
            'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
            'center': [center.x, center.y, center.z],
            'extent': [extent.x, extent.y, extent.z],
            'world_cord': global_verts,
            'semantic_tags': [npc.semantic_tags],
            'color': npc.attributes['color'],
            'speed': ego_speed,
            'brake': ego_brake,
            'road_id': road_id,
            'lane_id': lane_id,
            'section_id': section_id,
            'world2ego': world2ego,
        }
        results.append(result)

        # vehicles.vehicle
        vehicles = self.get_actor_filter_vehicle()
        for npc in vehicles.vehicle:
            if not npc.is_alive: continue
            if npc.id == self._vehicle.id: continue
            npc_id = str(npc.id)
            location = npc.get_transform().location
            rotation = npc.get_transform().rotation
            road_id = CarlaDataProvider.get_map().get_waypoint(location).road_id
            lane_id = CarlaDataProvider.get_map().get_waypoint(location).lane_id
            section_id = CarlaDataProvider.get_map().get_waypoint(location).section_id
            # verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
            # center, extent = get_center_and_extent(verts)
            # # from carla official
            # bb_cords = _bounding_box_to_world(npc.bounding_box)
            # world_cord = _vehicle_to_world(bb_cords, npc)
            # #
            # from handcraft
            world2vehicle = npc.get_transform().get_inverse_matrix()
            extent = npc.bounding_box.extent
            center = npc.get_transform().transform(npc.bounding_box.location)
            local_verts = calculate_cube_vertices(npc.bounding_box.location, npc.bounding_box.extent)
            global_verts = []
            for l_v in local_verts:
                g_v = npc.get_transform().transform(carla.Location(l_v[0], l_v[1], l_v[2]))
                global_verts.append([g_v.x, g_v.y, g_v.z])
            ###################
            vehicle_speed = self._get_forward_speed(transform=npc.get_transform(), velocity=npc.get_velocity())
            vehicle_brake = npc.get_control().brake
            vehicle_matrix = np.array(npc.get_transform().get_matrix())
            yaw = np.deg2rad(rotation.yaw)
            try:
                light_state = str(npc.get_light_state()).split('.')[-1]
            except:
                light_state = 'None'
            # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = get_relative_transform(ego_matrix, vehicle_matrix)
            if not lidar is None:
                num_in_bbox_lidar_points = self.get_lidar_points_in_bbox(relative_pos, relative_yaw, extent, lidar)
            else:
                num_in_bbox_lidar_points = -1
            
            distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
            result = {
                'class': 'vehicle',
                'state': 'dynamic',
                'id': npc_id,
                'location': [location.x, location.y, location.z],
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
                'center': [center.x, center.y, center.z],
                'extent': [extent.x, extent.y, extent.z],
                'world_cord': global_verts,
                'semantic_tags': npc.semantic_tags,
                'type_id': npc.type_id,
                'color': npc.attributes['color'] if 'color' in npc.attributes else None,
                'base_type': npc.attributes['base_type'],
                'num_points': int(num_in_bbox_lidar_points),
                'distance': distance,
                'speed': vehicle_speed,
                'brake': vehicle_brake,
                'light_state': light_state,
                'road_id': road_id,
                'lane_id': lane_id,
                'section_id': section_id,
                'world2vehicle': world2vehicle,
                # 'actor': npc, # for debug 
              }
            results.append(result)
        
        # vehicles.static
        car_bbox_list = self._world.get_level_bbs(carla.CityObjectLabel.Car) 
        bicycle_list = self._world.get_level_bbs(carla.CityObjectLabel.Bicycle)
        bus_list = self._world.get_level_bbs(carla.CityObjectLabel.Bus)
        motorcycle_list = self._world.get_level_bbs(carla.CityObjectLabel.Motorcycle)
        train_list = self._world.get_level_bbs(carla.CityObjectLabel.Train)
        truck_list = self._world.get_level_bbs(carla.CityObjectLabel.Truck)
        vehicles_static_bbox = car_bbox_list + bicycle_list + bus_list + motorcycle_list + train_list + truck_list
        vehicles_static_bbox_nearby = []
        for v_s in vehicles_static_bbox:
            if compute_2d_distance(v_s.location, self._vehicle.get_transform().location) < (DIS_LIGHT_SAVE + 20):
                vehicles_static_bbox_nearby.append(v_s)
        for npc in vehicles.static:
            if not npc.is_alive: continue
            new_bbox = None
            min_dis = 50
            for vehicle_bbox in vehicles_static_bbox_nearby:
                dis = compute_2d_distance(npc.get_transform().location, vehicle_bbox.location)
                if dis < min_dis:
                    new_bbox = vehicle_bbox
                    min_dis = dis
            if min_dis > 20:
                continue
            if not new_bbox:
                raise Exception('new_bbox is None')
            if new_bbox not in vehicles_static_bbox_nearby:
                raise Exception('new_bbox not in vehicles_static_bbox_nearby')
            vehicles_static_bbox_nearby.remove(new_bbox)
            npc_id = str(npc.id)
            # location = new_bbox.location
            # rotation = new_bbox.rotation
            # center = new_bbox.location
            extent = new_bbox.extent
            ####
            location = npc.get_transform().location
            rotation = npc.get_transform().rotation
            road_id = CarlaDataProvider.get_map().get_waypoint(location).road_id
            lane_id = CarlaDataProvider.get_map().get_waypoint(location).lane_id
            section_id = CarlaDataProvider.get_map().get_waypoint(location).section_id
            # verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
            # center, extent = get_center_and_extent(verts)
            # # from carla official
            # bb_cords = _bounding_box_to_world(npc.bounding_box)
            # world_cord = _vehicle_to_world(bb_cords, npc)
            # #
            # from handcraft
            world2vehicle = npc.get_transform().get_inverse_matrix()
            # extent = npc.bounding_box.extent
            # extent = carla.Vector3D(extent.y, extent.x, extent.z) # staic need swap
            center = npc.get_transform().transform(npc.bounding_box.location)
            local_verts = calculate_cube_vertices(npc.bounding_box.location, extent)
            global_verts = []
            for l_v in local_verts:
                g_v = npc.get_transform().transform(carla.Location(l_v[0], l_v[1], l_v[2]))
                # g_v = np.dot(np.matrix(npc.get_transform().get_inverse_matrix()).I, [l_v[0], l_v[1], l_v[2],1])
                global_verts.append([g_v.x, g_v.y, g_v.z])
            ###################
            vehicle_speed = self._get_forward_speed(transform=npc.get_transform(), velocity=npc.get_velocity())
            vehicle_brake = 1.0
            vehicle_matrix = np.array(npc.get_transform().get_matrix())
            yaw = np.deg2rad(rotation.yaw)
            light_state= 'NONE'
            # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
            relative_yaw = normalize_angle(yaw - ego_yaw)
            relative_pos = get_relative_transform(ego_matrix, vehicle_matrix)
            if not lidar is None:
                num_in_bbox_points = self.get_lidar_points_in_bbox(relative_pos, relative_yaw, extent, lidar)
            else:
                num_in_bbox_points = -1

            distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
            result = {
                'class': 'vehicle',
                'state': 'static',
                'id': npc_id,
                'location': [location.x, location.y, location.z],
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
                'center': [center.x, center.y, center.z],
                'extent': [extent.x, extent.y, extent.z],
                'world_cord': global_verts,
                'semantic_tags': npc.semantic_tags,
                'type_id': npc.attributes['mesh_path'],
                'num_points': int(num_in_bbox_points),
                'distance': distance,
                'speed': vehicle_speed,
                'brake': vehicle_brake,
                'light_state': light_state,
                'road_id': road_id,
                'lane_id': lane_id,
                'section_id': section_id,
                'world2vehicle': world2vehicle,
                # 'actor': npc, # for debug
                }
            results.append(result)
        
        # pedestrians
        pedestrians = self._world.get_actors().filter('walker*')
        for npc in pedestrians:
            if not npc.is_alive: 
                continue
            else:
                try:
                    if compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location) < DIS_WALKER_SAVE:
                        npc_id = str(npc.id)
                        location = npc.get_transform().location
                        rotation = npc.get_transform().rotation
                        road_id = CarlaDataProvider.get_map().get_waypoint(location).road_id
                        lane_id = CarlaDataProvider.get_map().get_waypoint(location).lane_id
                        section_id = CarlaDataProvider.get_map().get_waypoint(location).section_id
                        # verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
                        # center, extent = get_center_and_extent(verts)
                        # # from carla official
                        # bb_cords = _bounding_box_to_world(npc.bounding_box)
                        # world_cord = _vehicle_to_world(bb_cords, npc)
                        # #
                        # from handcraft
                        world2ped = npc.get_transform().get_inverse_matrix()
                        extent = npc.bounding_box.extent
                        center = npc.get_transform().transform(npc.bounding_box.location)
                        local_verts = calculate_cube_vertices(npc.bounding_box.location, npc.bounding_box.extent)
                        global_verts = []
                        for l_v in local_verts:
                            g_v = npc.get_transform().transform(carla.Location(l_v[0], l_v[1], l_v[2]))
                            global_verts.append([g_v.x, g_v.y, g_v.z])
                        ###################
                        walker_speed = self._get_forward_speed(transform=npc.get_transform(), velocity=npc.get_velocity())
                        # walker_speed = npc.attributes['speed'] #(TODO) yzj
                        walker_matrix = np.array(npc.get_transform().get_matrix())
                        yaw = np.deg2rad(rotation.yaw)
                        # bones_3d_lines = build_skeleton(npc, self.skeleton_links)
                        # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                        relative_yaw = normalize_angle(yaw - ego_yaw)
                        relative_pos = get_relative_transform(ego_matrix, walker_matrix)
                        if not lidar is None:
                            num_in_bbox_points = self.get_lidar_points_in_bbox(relative_pos, relative_yaw, extent, lidar)
                        else:
                            num_in_bbox_points = -1

                        distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
                        result = {
                            'class': 'walker',
                            'id': npc_id,
                            'location': [location.x, location.y, location.z],
                            'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                            'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
                            'center': [center.x, center.y, center.z],
                            'extent': [extent.x, extent.y, extent.z],
                            'world_cord': global_verts,
                            'semantic_tags': npc.semantic_tags,
                            'type_id': npc.type_id,
                            'gender': npc.attributes['gender'],
                            'age': npc.attributes['age'],
                            'num_points': int(num_in_bbox_points),
                            'distance': distance,
                            'speed': walker_speed,
                            # 'bone': bones_3d_lines,
                            'road_id': road_id,
                            'lane_id': lane_id,
                            'section_id': section_id,
                            'world2ped': world2ped,
                            # 'actor': npc, # for debug
                        }
                        results.append(result)
                except RuntimeError:
                    continue
        
        # traffic_light
        traffic_light = self.get_actor_filter_traffic_light()
        traffic_light_bbox = self._world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        traffic_light_bbox_nearby = []
        for light_bbox in traffic_light_bbox:
            if compute_2d_distance(light_bbox.location, self._vehicle.get_transform().location) < (DIS_LIGHT_SAVE + 20):
                traffic_light_bbox_nearby.append(light_bbox)
        for npc in traffic_light.lights:
            new_bbox = None
            min_dis = 50
            for light_bbox in traffic_light_bbox_nearby:
                dis = compute_2d_distance(npc.get_transform().location, light_bbox.location)
                if dis < min_dis:
                    new_bbox = light_bbox
                    min_dis = dis
            if min_dis > 20:
                continue
            traffic_light_bbox_nearby.remove(new_bbox)
            npc_id = str(npc.id)
            location = new_bbox.location
            rotation = new_bbox.rotation
            center = new_bbox.location
            extent = new_bbox.extent
            road_id = CarlaDataProvider.get_map().get_waypoint(new_bbox.location).road_id
            lane_id = CarlaDataProvider.get_map().get_waypoint(new_bbox.location).lane_id
            section_id = CarlaDataProvider.get_map().get_waypoint(new_bbox.location).section_id
            volume_location = npc.get_transform().transform(npc.trigger_volume.location)
            volume_rotation = carla.Rotation(pitch=(rotation.pitch + npc.trigger_volume.rotation.pitch)%360, roll=(rotation.roll + npc.trigger_volume.rotation.roll)%360, yaw=(rotation.yaw + npc.trigger_volume.rotation.yaw) % 360)
            state = npc.state
            distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
            if traffic_light.most_affect_light and str(traffic_light.most_affect_light.id) == npc_id:
                affects_ego = True
            else:
                affects_ego = False
            result = {
                'class': 'traffic_light',
                'id': npc_id,
                'location': [location.x, location.y, location.z],
                'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                'center': [center.x, center.y, center.z],
                'extent': [extent.x, extent.y, extent.z],
                'semantic_tags': npc.semantic_tags,
                'type_id': npc.type_id,
                'distance': distance,
                'state': state,
                'affects_ego': affects_ego,
                'trigger_volume_location': [volume_location.x, volume_location.y, volume_location.z],
                'trigger_volume_rotation': [volume_rotation.pitch, volume_rotation.roll, volume_rotation.yaw],
                'trigger_volume_extent': [npc.trigger_volume.extent.x, npc.trigger_volume.extent.y, npc.trigger_volume.extent.z],
                'road_id': road_id,
                'lane_id': lane_id,
                'section_id': section_id,
                # 'actor': npc, # for debug
                # 'new_bbox': new_bbox, # for debug
            }
            results.append(result)
        
        # traffic_sign
        traffic_sign = self.get_actor_filter_traffic_sign()
        traffic_sign_bbox = self._world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
        traffic_sign_bbox_nearby = []
        for sign_bbox in traffic_sign_bbox:
            if compute_2d_distance(sign_bbox.location, self._vehicle.get_transform().location) < (DIS_SIGN_SAVE + 20):
                traffic_sign_bbox_nearby.append(sign_bbox)

        for npc in traffic_sign.actors:
            if hasattr(npc, 'trigger_volume'):
                new_bbox = None
                min_dis = 50
                for sign_bbox in traffic_sign_bbox_nearby:
                    dis = compute_2d_distance(npc.get_transform().location, sign_bbox.location)
                    if dis < min_dis:
                        new_bbox = sign_bbox
                        min_dis = dis
                if min_dis > 20:
                    continue
                traffic_sign_bbox_nearby.remove(new_bbox)
                npc_id = str(npc.id)
                world2sign = npc.get_transform().get_inverse_matrix()
                location = new_bbox.location
                rotation = new_bbox.rotation
                center = new_bbox.location
                extent = new_bbox.extent
                road_id = CarlaDataProvider.get_map().get_waypoint(new_bbox.location).road_id
                lane_id = CarlaDataProvider.get_map().get_waypoint(new_bbox.location).lane_id
                section_id = CarlaDataProvider.get_map().get_waypoint(new_bbox.location).section_id
                volume_location = npc.get_transform().transform(npc.trigger_volume.location)
                volume_rotation = carla.Rotation(pitch=(rotation.pitch + npc.trigger_volume.rotation.pitch)%360, roll=(rotation.roll + npc.trigger_volume.rotation.roll)%360, yaw=(rotation.yaw + npc.trigger_volume.rotation.yaw) % 360)
                distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
                if traffic_sign.most_affect_sign and str(traffic_sign.most_affect_sign.id) == npc_id:
                    affects_ego = True
                else:
                    affects_ego = False
                result = {
                    'class': 'traffic_sign',
                    'id': npc_id,
                    'location': [location.x, location.y, location.z],
                    'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                    'center': [center.x, center.y, center.z],
                    'extent': [extent.x, extent.y, extent.z],
                    # 'extent': [extent.x, 0.5, extent.z],
                    'semantic_tags': npc.semantic_tags,
                    'type_id': npc.type_id,
                    'distance': distance,
                    'affects_ego': affects_ego,
                    'trigger_volume_location': [volume_location.x, volume_location.y, volume_location.z],
                    'trigger_volume_rotation': [volume_rotation.pitch, volume_rotation.roll, volume_rotation.yaw],
                    'trigger_volume_extent': [npc.trigger_volume.extent.x, npc.trigger_volume.extent.y, npc.trigger_volume.extent.z],
                    'road_id': road_id,
                    'lane_id': lane_id,
                    'section_id': section_id,
                    'world2sign': world2sign,
                    # 'actor': npc, # for debug
                    # 'new_bbox': new_bbox, # for debug
                }
            else:
                npc_id = str(npc.id)
                location = npc.get_transform().location
                rotation = npc.get_transform().rotation
                # verts = [v for v in npc.bounding_box.get_world_vertices(npc.get_transform())]
                # center, extent = get_center_and_extent(verts)
                # from handcraft
                world2sign = npc.get_transform().get_inverse_matrix()
                extent = npc.bounding_box.extent
                center = npc.get_transform().transform(npc.bounding_box.location)
                local_verts = calculate_cube_vertices(npc.bounding_box.location, npc.bounding_box.extent)
                global_verts = []
                for l_v in local_verts:
                    g_v = npc.get_transform().transform(carla.Location(l_v[0], l_v[1], l_v[2]))
                    global_verts.append([g_v.x, g_v.y, g_v.z])
                road_id = CarlaDataProvider.get_map().get_waypoint(location).road_id
                lane_id = CarlaDataProvider.get_map().get_waypoint(location).lane_id
                section_id = CarlaDataProvider.get_map().get_waypoint(location).section_id
                # distance = npc.get_transform().location.distance(self._vehicle.get_transform().location)
                distance = compute_2d_distance(npc.get_transform().location, self._vehicle.get_transform().location)
                if traffic_sign.most_affect_sign and str(traffic_sign.most_affect_sign.id) == npc_id:
                    affects_ego = True
                else:
                    affects_ego = False

                result = {
                    'class': 'traffic_sign',
                    'id': npc_id,
                    'location': [location.x, location.y, location.z],
                    'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
                    'bbx_loc': [npc.bounding_box.location.x, npc.bounding_box.location.y, npc.bounding_box.location.z],
                    'center': [center.x, center.y, center.z],
                    'extent': [extent.x, extent.y, extent.z],
                    'world_cord': global_verts,
                    # 'extent': [extent.x, 0.5, extent.z],
                    'semantic_tags': npc.semantic_tags,
                    'type_id': npc.type_id,
                    'distance': distance,
                    'affects_ego': affects_ego,
                    'road_id': road_id,
                    'lane_id': lane_id,
                    'section_id': section_id,
                    'world2sign': world2sign,
                    # 'actor': npc, # for debug
                }
            results.append(result)
        return results

    def polar_to_cartesian(self, altitude, azimuth, depth):
        """
        Convert polar coordinates (altitude, azimuth, depth) to Cartesian (x, y, z).
        Altitude and azimuth are assumed to be in radians.
        """
        z = depth * np.sin(altitude)
        r_cos_altitude = depth * np.cos(altitude)
        x = r_cos_altitude * np.cos(azimuth)
        y = r_cos_altitude * np.sin(azimuth)
        return x, y, z

    def get_radar_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, radar_data):
        """
        Checks for a given vehicle in ego coordinate system, how many RADAR hits there are in its bounding box.
        :param vehicle_pos: Relative position of the vehicle w.r.t. the ego [x, y, z]
        :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego in radians
        :param extent: List, half extent of the bounding box [length/2, width/2, height/2]
        :param radar_data: RADAR data with structure [altitude, azimuth, depth, velocity]
        :return: Returns the number of RADAR hits within the bounding box of the vehicle
        """
        radar_cartesian = np.array([self.polar_to_cartesian(np.radians(altitude), np.radians(azimuth), depth)
                                    for altitude, azimuth, depth, _ in radar_data])

        rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                    [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0],
                                    [0.0, 0.0, 1.0]])

        # Transform RADAR points to vehicle coordinate system
        vehicle_radar = (rotation_matrix.T @ (radar_cartesian - vehicle_pos).T).T

        # Half extents for the bounding box
        x, y, z = extent
        num_hits = ((vehicle_radar[:, 0] <= x) & (vehicle_radar[:, 0] >= -x) & 
                    (vehicle_radar[:, 1] <= y) & (vehicle_radar[:, 1] >= -y) & 
                    (vehicle_radar[:, 2] <= z) & (vehicle_radar[:, 2] >= -z)).sum()
        return num_hits
    
    def get_lidar_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, lidar):
        """
        Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
        :param vehicle_pos: Relative position of the vehicle w.r.t. the ego
        :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego
        :param extent: List, Extent of the bounding box
        :param lidar: LiDAR point cloud
        :return: Returns the number of LiDAR hits within the bounding box of the
        vehicle
        """

        rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                    [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])

        # LiDAR in the with the vehicle as origin
        vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T

        # check points in bbox
        x, y, z = extent.x, extent.y, extent.z
        num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
                    (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
        return num_points
    
    def gps_to_location(self, gps):
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
        pass
    
    def _get_latlon_ref(self):
        """
        Convert from waypoints world coordinates to CARLA GPS coordinates
        :return: tuple with lat and lon coordinates
        """
        xodr = CarlaDataProvider.get_map().to_opendrive()
        tree = ET.ElementTree(ET.fromstring(xodr))

        # default reference
        lat_ref = 42.0
        lon_ref = 2.0

        for opendrive in tree.iter("OpenDRIVE"):
            for header in opendrive.iter("header"):
                for georef in header.iter("geoReference"):
                    if georef.text:
                        str_list = georef.text.split(' ')
                        for item in str_list:
                            if '+lat_0' in item:
                                lat_ref = float(item.split('=')[1])
                            if '+lon_0' in item:
                                lon_ref = float(item.split('=')[1])
        return lat_ref, lon_ref

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None, plant=False):

        control = super().run_step(input_data, timestamp, plant=plant)

        tick_data = self.tick(input_data)

        if self.step % self.config.data_save_freq == 0:
            if self.save_path is not None and self.datagen:
                self.save_sensor(tick_data)
        return control