import carla
import json
import csv
import os
import sys
import random
import math
import numpy as np
from typing import Dict, List, Optional, Any
from utils import convert_depth
import pathlib
import laspy
import cv2
import time

WORKING_DIR = os.path.dirname(os.path.dirname(__file__))
CARLA_ROOT = "/nvme_data1/CarlaUE4/PythonAPI/carla"
sys.path.append(os.path.join(WORKING_DIR, "leaderboard_autopilot"))
sys.path.append(os.path.join(WORKING_DIR, "scenario_runner_autopilot"))
sys.path.append(os.path.join(CARLA_ROOT))
from leaderboard.envs.sensor_interface_local import (CallBack, SensorInterface)
VEHICLE_DIMENSIONS = [4.915, 1.937, 1.633]

class SensorManagement:
    def __init__(
        self,
        world,
        output_dir,
        save_depth = False,
        save_semantic = False,
    ):
        """
        Initialize the CARLA Sensor Management .
        
        Args:
            world: CARLA world
            port: CARLA server port
            output_dir: Directory to save output data
            save_depth: trigger CARLA depth sensors
            save_semantic: trigger CARLA semantic sensor
        """
        # Configuration
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.output_dir = pathlib.Path(output_dir)
        self.save_depth = save_depth
        self.save_semantic = save_semantic
        
        # camera params
        self.topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 150.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        self.unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
            
        self.sensors_list = []
        self.sensor_list_names = []
        self.sensors_mapping = {}
        self.sensor_interface = SensorInterface()
        self.make_save_dir()

    def make_save_dir(self):
        assert self.output_dir is not None, "Output directory is not specified"
        (self.output_dir).mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'camera').mkdir(exist_ok=True)
        (self.output_dir / 'lidar').mkdir(exist_ok=True)
        (self.output_dir / 'bev').mkdir(exist_ok=True)

        (self.output_dir / 'camera' / 'rgb_left').mkdir(exist_ok=True)
        (self.output_dir / 'camera' / 'rgb_front_left').mkdir(exist_ok=True)
        (self.output_dir / 'camera' / 'rgb_front_right').mkdir(exist_ok=True)
        (self.output_dir / 'camera' / 'rgb_right').mkdir(exist_ok=True)
        (self.output_dir / 'camera' / 'rgb_back_left').mkdir(exist_ok=True)
        (self.output_dir / 'camera' / 'rgb_back_right').mkdir(exist_ok=True)
        (self.output_dir / 'camera' / 'rgb_top_down').mkdir(exist_ok=True)
            
        if self.save_depth:
            (self.output_dir / 'camera' / 'depth_left').mkdir(exist_ok=True)
            (self.output_dir / 'camera' / 'depth_front_left').mkdir(exist_ok=True)
            (self.output_dir / 'camera' / 'depth_front_right').mkdir(exist_ok=True)
            (self.output_dir / 'camera' / 'depth_right').mkdir(exist_ok=True)
            (self.output_dir / 'camera' / 'depth_back_left').mkdir(exist_ok=True)
            (self.output_dir / 'camera' / 'depth_back_right').mkdir(exist_ok=True)

        if self.save_semantic:
            (self.output_dir / 'camera' / 'semantic_left').mkdir(exist_ok=True)
            (self.output_dir / 'camera' / 'semantic_front_left').mkdir(exist_ok=True)
            (self.output_dir / 'camera' / 'semantic_front_right').mkdir(exist_ok=True)
            (self.output_dir / 'camera' / 'semantic_right').mkdir(exist_ok=True)
            (self.output_dir / 'camera' / 'semantic_back_left').mkdir(exist_ok=True)
            (self.output_dir / 'camera' / 'semantic_back_right').mkdir(exist_ok=True)
        
    def _sensors_config(self):
        sensors = [
            # camera rgb
            {
                'type': 'sensor.camera.rgb',
                'x': -0.54,
                'y': -0.26,
                'z': VEHICLE_DIMENSIONS[2]/2.0+1.07,
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
                'z': VEHICLE_DIMENSIONS[2]/2.0+0.9,
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
                'z': VEHICLE_DIMENSIONS[2]/2.0+1.01,
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
                'z': VEHICLE_DIMENSIONS[2]/2.0+1.1,
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
                'z': VEHICLE_DIMENSIONS[2]/2.0+0.87,
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
                'z': VEHICLE_DIMENSIONS[2]/2.0+1.01,
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
                'x': -0.49, 'y': 0, 'z': VEHICLE_DIMENSIONS[2]/2.0+1.24,
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
            ### Debug sensor, not used by the model
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 50.0,
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 110,
                'id': 'TOP_DOWN'
                },
        ]
    
        if self.save_depth:
            depth_cameras = [
                {
                    'type': 'sensor.camera.depth',
                    'x': -0.54,
                    'y': -0.26,
                    'z': VEHICLE_DIMENSIONS[2]/2.0+1.07,
                    'roll': 90-89.98,
                    'pitch': -0.63,
                    'yaw': -72.23-90,
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
                    'z': VEHICLE_DIMENSIONS[2]/2.0+0.9,
                    'roll': 90-89.93,
                    'pitch': -0.18,
                    'yaw': -1.7-90,
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
                    'z': VEHICLE_DIMENSIONS[2]/2.0+1.01,
                    'roll': 90-88.5,
                    'pitch': -0.18,
                    'yaw': 69.11-90,
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
                    'z': VEHICLE_DIMENSIONS[2]/2.0+1.1,
                    'roll': 90-89.8,
                    'pitch': -0.25,
                    'yaw': 108.96-90,
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
                    'z': VEHICLE_DIMENSIONS[2]/2.0+0.87,
                    'roll': 90-90.42,
                    'pitch': -0.17,
                    'yaw': 179.8-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 86.93,
                    # 'fov': 61.75,
                    'id': 'CAM_RIGHT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 0.29,
                    'y': -0.22,
                    'z': VEHICLE_DIMENSIONS[2]/2.0+1.01,
                    'roll': 90-89.38,
                    'pitch': 0.68,
                    'yaw': -110.43-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 49.91,
                    # 'fov': 34.19, 
                    'id': 'CAM_BACK_RIGHT_DEPTH'
                    }]
            sensors.extend(depth_cameras)
        
        if self.save_semantic:
            semantic_cameras = [
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -0.54,
                    'y': -0.26,
                    'z': VEHICLE_DIMENSIONS[2]/2.0+1.07,
                    'roll': 90-89.98,
                    'pitch': -0.63,
                    'yaw': -72.23-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 52.12,
                    # 'fov': 36.24,
                    'id': 'CAM_BACK_LEFT_SEMANTIC'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -0.5,
                    'y': -0.7,
                    'z': VEHICLE_DIMENSIONS[2]/2.0+0.9,
                    'roll': 90-89.93,
                    'pitch': -0.18,
                    'yaw': -1.7-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 87.14,
                    # 'fov': 62.14,
                    'id': 'CAM_LEFT_SEMANTIC'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -0.24,
                    'y': -0.03,
                    'z': VEHICLE_DIMENSIONS[2]/2.0+1.01,
                    'roll': 90-88.5,
                    'pitch': -0.18,
                    'yaw': 69.11-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 53.13,
                    # 'fov': 36.83,
                    'id': 'CAM_FRONT_LEFT_SEMANTIC'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -0.43,
                    'y': -0.02,
                    'z': VEHICLE_DIMENSIONS[2]/2.0+1.1,
                    'roll': 90-89.8,
                    'pitch': -0.25,
                    'yaw': 108.96-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 52.21,
                    # 'fov': 36.02,
                    'id': 'CAM_FRONT_RIGHT_SEMANTIC'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -0.62,
                    'y': 0.64,
                    'z': VEHICLE_DIMENSIONS[2]/2.0+0.87,
                    'roll': 90-90.42,
                    'pitch': -0.17,
                    'yaw': 179.8-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 86.93,
                    # 'fov': 61.75,
                    'id': 'CAM_RIGHT_SEMANTIC'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': 0.29,
                    'y': -0.22,
                    'z': VEHICLE_DIMENSIONS[2]/2.0+1.01,
                    'roll': 90-89.38,
                    'pitch': 0.68,
                    'yaw': -110.43-90,
                    'width': 1920,
                    'height': 1080,
                    'fov': 49.91,
                    # 'fov': 34.19, 
                    'id': 'CAM_BACK_RIGHT_SEMANTIC'
                    }]
            sensors.extend(semantic_cameras)
            
        self.sensors_mapping = {}
        for sensor in sensors:
            self.sensors_mapping[sensor['id']] = sensor
        return sensors
    
    def _preprocess_sensor_spec(self, sensor_spec):
        type_ = sensor_spec["type"]
        id_ = sensor_spec["id"]
        attributes = {}

        if type_ == 'sensor.opendrive_map':
            attributes['reading_frequency'] = sensor_spec['reading_frequency']
            sensor_location = carla.Location()
            sensor_rotation = carla.Rotation()

        if type_ == 'sensor.camera.rgb':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])
            attributes['role_name'] = str(sensor_spec['id'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        
        elif type_ == 'sensor.camera.depth':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])
            attributes['role_name'] = str(sensor_spec['id'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                roll=sensor_spec['roll'],
                                                yaw=sensor_spec['yaw'])
        
        elif type_ == 'sensor.camera.semantic_segmentation':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])
            attributes['role_name'] = str(sensor_spec['id'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                roll=sensor_spec['roll'],
                                                yaw=sensor_spec['yaw'])
        
        elif type_ == 'sensor.camera.instance_segmentation':
            attributes['image_size_x'] = str(sensor_spec['width'])
            attributes['image_size_y'] = str(sensor_spec['height'])
            attributes['fov'] = str(sensor_spec['fov'])
            attributes['role_name'] = str(sensor_spec['id'])

            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                roll=sensor_spec['roll'],
                                                yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.lidar.ray_cast':
            attributes['range'] = str(sensor_spec['range'])
            attributes['rotation_frequency'] = str(sensor_spec['rotation_frequency'])
            attributes['channels'] = str(sensor_spec['channels'])
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['points_per_second'] = str(sensor_spec['points_per_second'])
            attributes['atmosphere_attenuation_rate'] = str(0.004)
            attributes['dropoff_general_rate'] = str(sensor_spec['dropoff_general_rate'])
            attributes['dropoff_intensity_limit'] = str(sensor_spec['dropoff_intensity_limit'])
            attributes['dropoff_zero_intensity'] = str(sensor_spec['dropoff_zero_intensity'])
            attributes['role_name'] = str(sensor_spec['id'])


            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.lidar.ray_cast_semantic':
            attributes['rotation_frequency'] = str(sensor_spec['rotation_frequency'])
            attributes['upper_fov'] = str(10)
            attributes['lower_fov'] = str(-30)
            attributes['points_per_second'] = str(sensor_spec['points_per_second'])
            attributes['role_name'] = str(sensor_spec['id'])


            sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])

        elif type_ == 'sensor.other.radar':
            attributes['horizontal_fov'] = str(sensor_spec['horizontal_fov'])  # degrees
            attributes['vertical_fov'] = str(sensor_spec['vertical_fov'])  # degrees
            attributes['points_per_second'] = '1500'
            attributes['range'] = sensor_spec['range']  # meters
            attributes['role_name'] = str(sensor_spec['id'])


            sensor_location = carla.Location(x=sensor_spec['x'],
                                             y=sensor_spec['y'],
                                             z=sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                             roll=sensor_spec['roll'],
                                             yaw=sensor_spec['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        return type_, id_, sensor_transform, attributes

    def setup_sensors(self, agent):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        for sensor_spec in self._sensors_config():  # Fixed: call the method
            type_, id_, sensor_transform, attributes = self._preprocess_sensor_spec(sensor_spec)

            bp = self.blueprint_library.find(type_)
            for key, value in attributes.items():
                bp.set_attribute(str(key), str(value))
            sensor = self.world.spawn_actor(bp, sensor_transform, agent)

            # setup callback
            sensor.listen(CallBack(id_, type_, sensor, self.sensor_interface))
            self.sensors_list.append(sensor)
            self.sensor_list_names.append([sensor_spec['id'], sensor])
            print(f"  - {type_} (id: {id_})")

        # Some sensors miss sending data during the first ticks, so tick several times and remove the data
        print('Warming up sensors...')
        for _ in range(10):
            self.world.tick()
        time.sleep(0.01)
    
    def get_data(self, frame):
        return self.sensor_interface.get_data(frame)    
            
    def save_sensor_data(self, tick_data, frame):
        # camera_bgr
        cam_bgr_left = tick_data['CAM_LEFT'][1][:, :, :3]
        cam_bgr_front_left = tick_data['CAM_FRONT_LEFT'][1][:, :, :3]
        cam_bgr_front_right = tick_data['CAM_FRONT_RIGHT'][1][:, :, :3]
        cam_bgr_right = tick_data['CAM_RIGHT'][1][:, :, :3]
        cam_bgr_back_left = tick_data['CAM_BACK_LEFT'][1][:, :, :3]
        cam_bgr_back_right = tick_data['CAM_BACK_RIGHT'][1][:, :, :3]
        cam_bgr_top_down = tick_data['TOP_DOWN'][1][:, :, :3]

        if self.save_depth:
            cam_bgr_left_depth = tick_data['CAM_LEFT_DEPTH'][1][:, :, :3]
            cam_bgr_front_left_depth = tick_data['CAM_FRONT_LEFT_DEPTH'][1][:, :, :3]
            cam_bgr_front_right_depth = tick_data['CAM_FRONT_RIGHT_DEPTH'][1][:, :, :3]
            cam_bgr_right_depth = tick_data['CAM_RIGHT_DEPTH'][1][:, :, :3]
            cam_bgr_back_left_depth = tick_data['CAM_BACK_LEFT_DEPTH'][1][:, :, :3]
            cam_bgr_back_right_depth = tick_data['CAM_BACK_RIGHT_DEPTH'][1][:, :, :3]

            # cam_gray_depth, 16 bit would be ideal, but we can't afford the extra storage.
            cam_gray_left_depth = convert_depth(cam_bgr_left_depth)
            cam_gray_front_left_depth = convert_depth(cam_bgr_front_left_depth)
            cam_gray_front_right_depth = convert_depth(cam_bgr_front_right_depth)
            cam_gray_right_depth = convert_depth(cam_bgr_right_depth)
            cam_gray_back_left_depth = convert_depth(cam_bgr_back_left_depth)
            cam_gray_back_right_depth = convert_depth(cam_bgr_back_right_depth)

        if self.save_semantic:
            cam_semantic_left = tick_data['CAM_LEFT_SEMANTIC'][1][:, :, :3]
            cam_semantic_front_left = tick_data['CAM_FRONT_LEFT_SEMANTIC'][1][:, :, :3]
            cam_semantic_front_right = tick_data['CAM_FRONT_RIGHT_SEMANTIC'][1][:, :, :3]
            cam_semantic_right = tick_data['CAM_RIGHT_SEMANTIC'][1][:, :, :3]
            cam_semantic_back_left = tick_data['CAM_BACK_LEFT_SEMANTIC'][1][:, :, :3]
            cam_semantic_back_right = tick_data['CAM_BACK_RIGHT_SEMANTIC'][1][:, :, :3]
        
        # CARLA images are already in opencv's BGR format.
        cv2.imwrite(str(self.output_dir / 'camera' / 'rgb_left' / (f'{frame:05}.jpg')), cam_bgr_left, [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.output_dir / 'camera' / 'rgb_front_left' / (f'{frame:05}.jpg')), cam_bgr_front_left, [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.output_dir / 'camera' / 'rgb_front_right' / (f'{frame:05}.jpg')), cam_bgr_front_right, [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.output_dir / 'camera' / 'rgb_right' / (f'{frame:05}.jpg')), cam_bgr_right, [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.output_dir / 'camera' / 'rgb_back_left' / (f'{frame:05}.jpg')), cam_bgr_back_left, [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.output_dir / 'camera' / 'rgb_back_right' / (f'{frame:05}.jpg')), cam_bgr_back_right, [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(self.output_dir / 'camera' / 'rgb_top_down' / (f'{frame:05}.jpg')), cam_bgr_top_down, [cv2.IMWRITE_JPEG_QUALITY, 20])

        if self.save_depth:
            cv2.imwrite(str(self.output_dir / 'camera' / 'depth_left' / (f'{frame:05}.png')), cam_gray_left_depth)
            cv2.imwrite(str(self.output_dir / 'camera' / 'depth_front_left' / (f'{frame:05}.png')), cam_gray_front_left_depth)
            cv2.imwrite(str(self.output_dir / 'camera' / 'depth_front_right' / (f'{frame:05}.png')), cam_gray_front_right_depth)
            cv2.imwrite(str(self.output_dir / 'camera' / 'depth_right' / (f'{frame:05}.png')), cam_gray_right_depth)
            cv2.imwrite(str(self.output_dir / 'camera' / 'depth_back_left' / (f'{frame:05}.png')), cam_gray_back_left_depth)
            cv2.imwrite(str(self.output_dir / 'camera' / 'depth_back_right' / (f'{frame:05}.png')), cam_gray_back_right_depth)
        
        if self.save_semantic:
            cv2.imwrite(str(self.output_dir / 'camera' / 'semantic_left' / (f'{frame:05}.jpg')), cam_semantic_left, [cv2.IMWRITE_JPEG_QUALITY, 20])
            cv2.imwrite(str(self.output_dir / 'camera' / 'semantic_front_left' / (f'{frame:05}.jpg')), cam_semantic_front_left, [cv2.IMWRITE_JPEG_QUALITY, 20])
            cv2.imwrite(str(self.output_dir / 'camera' / 'semantic_front_right' / (f'{frame:05}.jpg')), cam_semantic_front_right, [cv2.IMWRITE_JPEG_QUALITY, 20])
            cv2.imwrite(str(self.output_dir / 'camera' / 'semantic_right' / (f'{frame:05}.jpg')), cam_semantic_right, [cv2.IMWRITE_JPEG_QUALITY, 20])
            cv2.imwrite(str(self.output_dir / 'camera' / 'semantic_back_left' / (f'{frame:05}.jpg')), cam_semantic_back_left, [cv2.IMWRITE_JPEG_QUALITY, 20])
            cv2.imwrite(str(self.output_dir / 'camera' / 'semantic_back_right' / (f'{frame:05}.jpg')), cam_semantic_back_right, [cv2.IMWRITE_JPEG_QUALITY, 20])

            
        # Specialized LiDAR compression format
        lidar = tick_data['LIDAR_TOP'][1][:, :3]
        header = laspy.LasHeader(point_format=0)  # LARS point format used for storing
        header.offsets = np.min(lidar, axis=0)
        point_precision = 0.001
        header.scales = np.array([point_precision, point_precision, point_precision])

        with laspy.open(self.output_dir / 'lidar' / (f'{frame:05}.laz'), mode='w', header=header) as writer:
            point_record = laspy.ScaleAwarePointRecord.zeros(lidar.shape[0], header=header)
            point_record.x = lidar[:, 0]
            point_record.y = lidar[:, 1]
            point_record.z = lidar[:, 2]
            writer.write_points(point_record)

    def cleanup(self):
        """Clean up all resources and destroy all actors."""
        print("Cleaning up...")
        
        # Destroy sensors
        for sensor in self.sensors_list:
            try:
                sensor.stop()  # Stop sensor listening
                sensor.destroy()
            except:
                pass  # Sensor might already be destroyed
        self.sensors_list.clear()
        self.sensor_list_names.clear()
        del self.sensor_interface
        
        print("Sensor Management Cleanup completed!")
