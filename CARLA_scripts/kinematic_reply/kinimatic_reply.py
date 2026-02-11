import carla
import json
import csv
import os
import sys
import random
import math
import numpy as np
from typing import Dict, List, Optional, Any
from utils import (gps_to_carla_location,
                   build_annotation_ref_json,
                   CARLA_VEHICLE_TYPES,
                   CARLA_MOTOCYCLE_TYPES,
                   CARLA_BICYCLE_TYPES)
import pathlib
import time
from sensor_management import SensorManagement
import argparse


WORKING_DIR = os.path.dirname(os.path.dirname(__file__))
CARLA_ROOT = "/nvme_data1/CarlaUE4/PythonAPI/carla"
sys.path.append(os.path.join(WORKING_DIR, "leaderboard_autopilot"))
sys.path.append(os.path.join(WORKING_DIR, "scenario_runner_autopilot"))
sys.path.append(os.path.join(CARLA_ROOT))


class CarlaReplaySimulator:
    def __init__(
        self,
        port: int = 12345,
        client_timeout: float = 300.0,
        tick_rate: float = 10,  # Hz
        dataset_root: str = "DrivIng",
        output_dir: str = "output",
        save_depth: bool = False,
        save_semantic: bool = False,
        start_timestamp: Optional[int] = -1,
        total_frames: Optional[int] = 200,
        debug: bool = False,
        initial_offset: Optional[carla.Vector3D] = None,
    ):
        """
        Initialize the CARLA Replay Simulator.
        
        Args:
            host: CARLA server host
            port: CARLA server port
            client_timeout: Timeout for client operations (seconds)
            tick_rate: Simulation frequency (Hz)
            dataset_root: Path to annotations.json
            timesync_file: Path to timesync_info.csv
            vehicle_state_dir: Directory containing vehicle_state JSON files
            output_dir: Directory to save output data
            skip_tracks: List of track IDs to skip
        """
        # Configuration
        self.host = 'localhost'
        self.port = port
        self.client_timeout = client_timeout
        self.tick_rate = tick_rate
        self.dataset_root = dataset_root
        self.annotation_file = os.path.join(self.dataset_root, "annotations_ref.json")
        self.timesync_file = os.path.join(self.dataset_root, "timesync_info.csv")
        self.vehicle_state_dir = os.path.join(self.dataset_root, "vehicle_state")
        self.calib_file = os.path.join(self.dataset_root, "calibration.json")
        self.output_dir = pathlib.Path(output_dir) / str(start_timestamp)
        self.save_depth = save_depth
        self.save_semantic = save_semantic

        # Initialize CARLA objects
        self.town = "Grid0828"
        self.client = None
        self.world = None
        self.map = None
        self.blueprint_library = None
        self.spectator = None
        self.sensor_management = None
        self.debug = debug
        
        # Loaded data
        self.scene_info = {}
        self.timestamps = []
        self._load_dataset()
       
        # Runtime state
        self.start_frame_idx = self.timestamps.index(start_timestamp) if start_timestamp in self.timestamps else 0
            
        self.end_frame_idx = min(self.start_frame_idx + total_frames, len(self.timestamps))
        self.current_frame_idx = 0
        assert self.start_frame_idx < self.end_frame_idx, \
            "End frame exceeds available frames"
            
        self.ego_vehicle = None
        self.initial_offset = initial_offset if initial_offset is not None else carla.Vector3D(x=0, y=0, z=0.0)
        self.active_actors = {}
        
        self.is_running = False

    def _load_dataset(self):
        """Load annotation data from JSON file."""
        if not os.path.exists(self.annotation_file):
            print(f"Geo-referenced annotation file not found: {self.annotation_file}. Building from raw data...")
            pre_anno = os.path.join(self.dataset_root, "annotations.json")
            assert os.path.exists(pre_anno), f"Annotation file not found: {self.annotation_file}"
            build_annotation_ref_json(pre_anno, self.timesync_file, self.vehicle_state_dir, self.calib_file, self.annotation_file)
        
        with open(self.annotation_file, 'r') as f:
            self.scene_info = json.load(f)
        self.timestamps = self.scene_info["timestamps"]
        print(f"Loaded scene info for {len(self.timestamps)} timestamps")
    
    def _spawn_agent(self, agent: Dict):
        """Spawn a new agent based on agent data."""
        track_id = agent['track_id']
        obj_type = agent['object_type']
        world_transform = self._update_npc_transform(agent)

        # 2. Determine Blueprint Filter
        if "Car" == obj_type:
            bp_filter = random.choice(CARLA_VEHICLE_TYPES)
        elif "Truck" == obj_type:
            bp_filter = "vehicle.carlamotors.european_hgv"
        elif "Van" == obj_type:
            bp_filter = "vehicle.mercedes.sprinter"
        elif "Pedestrian" in obj_type:
            bp_filter = "walker.pedestrian.*"
        elif "Cyclist" == obj_type:
            bp_filter = random.choice(CARLA_BICYCLE_TYPES)
        elif "Motorcycle" == obj_type:
            bp_filter = random.choice(CARLA_MOTOCYCLE_TYPES)
        else:
            return None  # Unsupported type

        blueprints = self.blueprint_library.filter(bp_filter)
        if not blueprints:
            print(f"Warning: No blueprint found for class '{obj_type}'")
            return

        bp = random.choice(blueprints)

        try:
            world_transform.location.z += 200 
            actor = self.world.spawn_actor(bp, world_transform)
            actor.set_simulate_physics(False)
            world_transform.location.z -= 200
            actor.set_transform(world_transform)

            self.active_actors[track_id] = actor
            if self.debug:
                print(f"[DEBUG] Spawned {obj_type} {track_id} - {world_transform.location}")
        except Exception as e:
            print(f"WARNING: Failed to spawn actor for track {track_id}: {e}")

    def _teleport_agent(self, agent: Dict):
        """Teleport an existing agent to new position."""
        track_id = agent['track_id']
        if track_id not in self.active_actors:
            return  # Shouldn't happen, but safety check
        actor = self.active_actors[track_id]
        world_transform = self._update_npc_transform(agent)
        try:
            actor.set_transform(world_transform)
            if self.debug:
                print(f"[DEBUG] Teleported {track_id} - {world_transform.location}")
        except Exception as e:
            print(f"WARNING: {e} {track_id}")
            
    def _delete_agent(self, track_id: str):
        """Delete an existing agent."""
        if track_id in self.active_actors:
            actor = self.active_actors[track_id]
            actor.destroy()
            del self.active_actors[track_id]
            if self.debug:
                print(f"[DEBUG] Removed actor {track_id}")
            
    def _initialize_ego_vehicle(self):  # Added: missing method
        """Initialize ego vehicle."""
        print("Initializing ego vehicle...")
        
        # Get initial ego state from first frame
        transform = self._update_ego_transform(self.start_frame_idx, True)
        initial_timestamp = self.timestamps[self.start_frame_idx]
        
        # Spawn ego vehicle
        blueprint = self.blueprint_library.filter('vehicle.*')[0]  # Use first available vehicle
        blueprint.set_attribute('role_name', 'hero')
        
        self.ego_vehicle = self.world.spawn_actor(blueprint, transform)
        control = carla.VehicleControl(hand_brake=True, brake=1.0)
        self.ego_vehicle.apply_control(control)
        self.ego_vehicle.set_simulate_physics(False)
        
        # Warm up the simulation
        for _ in range(5):
            self.world.tick()
        time.sleep(0.01)

        # Setup sensors
        print("Setting up sensors...")
        self.sensor_management.setup_sensors(self.ego_vehicle)
        print(f"Setup {len(self.sensor_management.sensors_list)} sensors")
        print(f"Starting timestamp : {initial_timestamp}, frame idx: {self.start_frame_idx}")
    
    def _update_npc_transform(self, agent):
        pos = agent['position']  # [x, y, z] relative to ego
        dim = agent['dimension']  # [length, width, height]
        
        x, y = gps_to_carla_location(pos[1], pos[0])
        x -= self.initial_offset.x
        y -= self.initial_offset.y
        
        z = pos[2] - self.initial_offset.z - dim[2]/2 # slight offset to avoid spawning inside the ground
        yaw = agent['azimuth'] + 180
        location = carla.Location(x=x, y=y, z=z+0.5)
        new_location = self.world.ground_projection(location)
        if new_location is None:
            new_location = location
            new_location.z -= 0.5 
        else:
            new_location = new_location.location

        return carla.Transform(location=new_location,
                               rotation=carla.Rotation(yaw=yaw))
    
    def _update_ego_transform(self, frame_idx, initial=False):
        ego_state = self.scene_info["ego_positions"][frame_idx]
        x, y = gps_to_carla_location(ego_state[1], ego_state[0])
        
        if initial:
            location = carla.Location(x, y, ego_state[2]-200)
        else:
            location = carla.Location(x=x, y=y, z=ego_state[2]+0.5)
            new_location = self.world.ground_projection(location)
            if new_location is None:
                new_location = location
                new_location.z -= 0.5 
            else:
                new_location = new_location.location
            location = new_location - self.initial_offset 
            
        transform = carla.Transform(location=location,
                                    rotation=carla.Rotation(yaw=-ego_state[3] - 90))

        # Move spectator to initial position
        self.spectator.set_transform(carla.Transform(
            carla.Location(x=transform.location.x, y=transform.location.y, z=transform.location.z + 50),
            carla.Rotation(pitch=-90, yaw=transform.rotation.yaw, roll=0)
        ))
        return transform
        
    def setup_environment(self):
        """Setup CARLA environment efficiently by skipping redundant configurations."""
        print("Initializing CARLA environment...")
        
        # 1. Connect to CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.client_timeout)
        
        # 2. Get current world and check map
        # We fetch the world first to check the current map name
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        current_map = self.map.name.split('/')[-1] # Extracts 'Town01' from full path
        
        if current_map != self.town:
            print(f"Map mismatch (Current: {current_map}, Desired: {self.town}). Loading new map...")
            self.world = self.client.load_world(self.town)
        else:
            print(f"Map '{self.town}' is already loaded. Skipping load_world.")

        # 3. Check and apply synchronous settings
        settings = self.world.get_settings()
        desired_delta = 1.0 / self.tick_rate

        # Handle NoneType safely
        current_delta = settings.fixed_delta_seconds if settings.fixed_delta_seconds is not None else 0.0

        # Only apply if there's a change to avoid a world-reset lag
        needs_sync_change = settings.synchronous_mode != True
        needs_delta_change = abs(current_delta - desired_delta) > 0.0001

        if needs_sync_change or needs_delta_change:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = desired_delta
            self.world.apply_settings(settings)
            print(f"Settings updated: sync=True, delta={desired_delta}s")
        else:
            print("Settings already match. Skipping apply_settings.")

        # 4. Finalize
        self.blueprint_library = self.world.get_blueprint_library()
        self.spectator = self.world.get_spectator()
        self.is_running = True
        self.sensor_management = SensorManagement(self.world, self.output_dir, self.save_depth, self.save_semantic)
        self._initialize_ego_vehicle()

    def tick(self) -> bool:
        """
        Advance simulation by one frame and save data.
        
        Returns:
            True if there are more frames to process, False otherwise
        """
        if not self.is_running or self.current_frame_idx + self.start_frame_idx >= self.end_frame_idx:
            print("No more frames to process or simulator not running.")
            return False

        frame_timestamp = self.timestamps[self.current_frame_idx + self.start_frame_idx]
        agents_for_current_frame = self.scene_info["data"][self.current_frame_idx + self.start_frame_idx]
        
        # Teleport ego vehicle to new position
        ego_curr_trans = self.ego_vehicle.get_transform()
        ego_curr_trans.location.z -= 200  # temporarily move underground to avoid collision during teleportation
        self.ego_vehicle.set_transform(ego_curr_trans)
        self.world.tick()  # Advance simulation to apply z-offset teleportation
        time.sleep(0.01)
        new_ego_transform = self._update_ego_transform(self.current_frame_idx + self.start_frame_idx)
        self.ego_vehicle.set_transform(new_ego_transform)
        if self.debug:
            print(f"[DEBUG] Teleoport ego vehicle {new_ego_transform.location}")

        # Get agents for current frame
        current_track_ids = {agent['track_id'] for agent in agents_for_current_frame}
        existing_track_ids = set(self.active_actors.keys())
        
        # set all the vehicle underground to avoid collision during teleportation
        for track_id in list(existing_track_ids):
            actor = self.active_actors[track_id]
            ori_transform = actor.get_transform()
            ori_transform.location.z -= 200
            actor.set_transform(ori_transform)
        self.world.tick()  # Advance simulation to apply z-offset teleportation
        
        agents_to_teleport = current_track_ids.intersection(existing_track_ids)
        for agent in agents_for_current_frame:
            track_id = agent['track_id']
            if track_id in agents_to_teleport:
                self._teleport_agent(agent)

        # Agents to spawn (in current frame but not in previous frame)
        agents_to_spawn = current_track_ids - existing_track_ids
        for agent in agents_for_current_frame:
            track_id = agent['track_id']
            if track_id in agents_to_spawn:
                self._spawn_agent(agent)
        
        # Advance simulation to make sure all changes take effect
        for _ in range(2):
            self.world.tick()
        time.sleep(0.01)  # Small delay to ensure stability
            
        game_frame = self.world.get_snapshot().frame
        input_data = self.sensor_management.get_data(game_frame)
        self.sensor_management.save_sensor_data(input_data, self.current_frame_idx)
        print(f"Simulation advanced. current frame {self.current_frame_idx + self.start_frame_idx}")

        # Move to next frame
        self.current_frame_idx += 1
        
        # Return whether there are more frames
        return self.current_frame_idx + self.start_frame_idx < self.end_frame_idx

    def cleanup(self):
        """Clean up all resources and destroy all actors."""
        print("Cleaning up...")
        
        # Destroy sensors
        self.sensor_management.cleanup()
        
        # Destroy ego vehicle
        if self.ego_vehicle:
            try:
                self.ego_vehicle.destroy()
            except:
                pass
            self.ego_vehicle = None
        
        # Destroy other actors
        for actor in self.active_actors.values():
            try:
                actor.destroy()
            except:
                pass
        self.active_actors.clear()
        self.spectator = None
        
        # Reset runtime state
        self.current_frame_idx = 0
        self.is_running = False
        
        # reset world settings to non-synchronous
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        print("Cleanup completed!")

def get_args():
    parser = argparse.ArgumentParser(description="Process dataset for DrivIng.")
    parser.add_argument(
        "--port", 
        type=int, 
        default=2000,
        help="Port of the CARLA server"
    )
    parser.add_argument(
        "--dataset_root", 
        type=str, 
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Directory to save output"
    )
    parser.add_argument(
        "--save_depth", 
        action="store_true", 
        help="If provided, depth maps will be saved"
    )
    parser.add_argument(
        "--save_semantic", 
        action="store_true", 
        help="If provided, semantic maps will be saved"
    )
    parser.add_argument(
        "--start_timestamp", 
        type=int, 
        default=1750163981799976000, 
        help="The starting timestamp or frame index (default: 100)"
    )
    parser.add_argument(
        "--total_frames", 
        type=int, 
        default=200, 
        help="Total number of frames to process (default: 200)"
    )

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_args()
    
    simulator = CarlaReplaySimulator(
        port=args.port,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        save_depth=args.save_depth,
        save_semantic=args.save_semantic,
        start_timestamp=args.start_timestamp,
        total_frames=args.total_frames,
    )
    
    try:
        # Setup environment
        simulator.setup_environment()
        while simulator.tick():
            pass  # Continue until all frames are processed
        
    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always clean up
        simulator.cleanup()