#!/usr/bin/env python
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Pedestrians crossing through the middle of the lane.
"""

from __future__ import print_function

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      KeepVelocity,
                                                                      WaitForever,
                                                                      Idle,
                                                                    #   StartAIWalkerControll,
                                                                    #   BatchStopAIWalkerControll,
                                                                      ActorTransformSetter,
                                                                      MovePedestrianWithEgo)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance)
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.tools.background_manager import HandleJunctionScenario
import numpy as np
import random


def convert_dict_to_location(actor_dict):
    """
    Convert a JSON string to a Carla.Location
    """
    location = carla.Location(
        x=float(actor_dict['x']),
        y=float(actor_dict['y']),
        z=float(actor_dict['z'])
    )
    return location

def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default



class NatualPedestrian(BasicScenario):

    """
    This class holds everything required for a group of natual pedestrians.
    And encounters a group of pedestrians.

    This is a single ego vehicle scenario.

    Notice that the initial pedestrian will walk from the start of the junction ahead to end_walker_flow_1.
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._world = CarlaDataProvider.get_world()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._wmap.get_waypoint(self._trigger_location)
        self._rng = CarlaDataProvider.get_random_seed()
        self._num_walkers = get_value_parameter(config, "num_walkers", int, 40)
        self._ego_end_distance = get_value_parameter(config, "ego_end_distance", int, 50)
        self._spawn_radius = get_value_parameter(config, "spawn_radius", int, 100)

        self.timeout = timeout
        self.walker_controllers = []
        self.walker_data = []

        super().__init__("NatualPedestrian",
                          ego_vehicles,
                          config,
                          world,
                          debug_mode,
                          criteria_enable=criteria_enable)

    def _initialize_actors(self, config):

        # Get the start point of the initial pedestrian

        # Spawn the walkers
        ego_location = self._reference_waypoint.transform.location
        walker_controller_bp = self._world.get_blueprint_library().find('controller.ai.walker')
        
        trials = self._num_walkers * 2
        radius_factor = np.sqrt(np.random.rand(trials))
        angle = 2.0 * np.pi * np.random.rand(trials)
        radius = self._spawn_radius * radius_factor
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)
        
        spawned_walker = 0
        
        for i in range(trials):
            if spawned_walker > self._num_walkers:
                break
            
            spawn_location = carla.Location(
                ego_location.x + x_offset[i],
                ego_location.y + y_offset[i],
                ego_location.z + 1.0 # A small Z offset to avoid spawning under the map
            )
            
            waypoint = self._wmap.get_waypoint(spawn_location, project_to_road=True, lane_type=carla.LaneType.Sidewalk)
            destination = self._world.get_random_location_from_navigation()
            if waypoint is not None and destination is not None:
                spawn_point = waypoint.transform
                spawn_point.location.z += 1.0
                
                walker = CarlaDataProvider.request_new_actor('walker.*', spawn_point)
                if walker is not None:
                    walker.set_location(spawn_point.location)
                    # walker = self._replace_walker(walker)
                    # ai_controller.start()
                    # ai_controller.go_to_location(destination)
                    walker_speed = 1 + random.random()
                    # ai_controller.set_max_speed(walker_speed)
                    # self.controller_ai_walker.append(ai_controller)
                    self.other_actors.append(walker)
                    self.walker_data.append({"transform": spawn_point,
                                             "destination": destination,
                                            "speed": walker_speed,
                                            "walker_id": walker.id})
                    spawned_walker += 1
        
        # SpawnActor = carla.command.SpawnActor
        # batch = []
        # for walker in self.other_actors:
        #     batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walker.id))
        # results = CarlaDataProvider.get_client().apply_batch_sync(batch, True)
        # for i in range(len(results)):
        #     if not results[i].error:
        #         self.walker_controllers.append(results[i].actor_id)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """
        sequence = py_trees.composites.Sequence(name="NatualPedestrian")
        sequence.add_child(Idle(0.5))
        # sequence.add_child(StartAIWalkerControll(self.walker_controllers, self.walker_data))      
        
        main_behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="WalkerMovement")

        main_behavior.add_child(DriveDistance(self.ego_vehicles[0], self._ego_end_distance, name="EndCondition"))
        sequence.add_child(main_behavior)

        # Remove everything
        # sequence.add_child(BatchStopAIWalkerControll(self.walker_controllers))
        for actor in self.other_actors:
            sequence.add_child(ActorDestroy(actor))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]
    
    def remove_all_actors(self):
        """
        Remove all actors
        """
        for i, _ in enumerate(self.other_actors):
            if self.other_actors[i] is not None:
                if CarlaDataProvider.actor_id_exists(self.other_actors[i].id):
                    CarlaDataProvider.remove_actor_by_id(self.other_actors[i].id)
                self.other_actors[i] = None
        self.other_actors = []
        
        controllers = self._world.get_actors(self.walker_controllers)
        for walker_controll in controllers:
            walker_controll.stop()
            walker_controll.destroy()
        self.controller_walker_ids = []

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

    # TODO: Pedestrian have an issue with large maps were setting them to dormant breaks them,
    # so all functions below are meant to patch it until the fix is done
    def _replace_walker(self, walker):
        """As the adversary is probably, replace it with another one"""
        type_id = walker.type_id
        walker.destroy()
        spawn_transform = self.ego_vehicles[0].get_transform()
        spawn_transform.location.z -= 50
        walker = CarlaDataProvider.request_new_actor(type_id, spawn_transform)
        if not walker:
            raise ValueError("Couldn't spawn the walker substitute")
        walker.set_simulate_physics(True)
        walker.set_location(spawn_transform.location)
        return walker

    # def _setup_scenario_trigger(self, config):
    #     """Normal scenario trigger but in parallel, a behavior that ensures the pedestrian stays active"""
    #     trigger_tree = super()._setup_scenario_trigger(config)

    #     if not self.route_mode:
    #         return trigger_tree

    #     parallel = py_trees.composites.Parallel(
    #         policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="ScenarioTrigger")

    #     for i, walker in enumerate(reversed(self.other_actors)):
    #         parallel.add_child(MovePedestrianWithEgo(self.ego_vehicles[0], walker, 100))

    #     parallel.add_child(trigger_tree)
    #     return parallel
