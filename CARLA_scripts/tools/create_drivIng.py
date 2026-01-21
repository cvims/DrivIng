import carla
from pprint import pp
import xml.etree.ElementTree as ET
from copy import deepcopy
import random
import xmltodict
import os
from leaderboard_autopilot.leaderboard.utils.route_manipulation import *
from agents.navigation.global_route_planner import GlobalRoutePlanner
from networkx.exception import NetworkXNoPath

route_config = {"@id": 0,
                "@town":None,
                "weathers" : {"weather" : []},
                "waypoints": {"position" : []},
                "scenarios": [
                {'scenario' : {"@name": None,
                "@type": 'NatualPedestrian',
                "trigger_point":None,
                "spawn_radius" : {"@value": "100"},
                "num_walkers": {"@value": "100"}}},]}

def path_exist(grp, waypoints_trajectory):
    for i in range(len(waypoints_trajectory) - 1):
        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        try:
            grp.trace_route(waypoint.transform.location, waypoint_next.transform.location)
        except NetworkXNoPath:
            print(f"not path existed, skip the path")
            return False
    return True

def draw_lane_points(carla_map, road_id, lane_id):
    all_waypoints = carla_map.generate_waypoints(5.0)
    target_lane_waypoints = [w for w in all_waypoints if w.road_id == road_id and w.lane_id == lane_id]
    spectator = world.get_spectator()
    transform = target_lane_waypoints[0].transform
    transform.location.z = 100
    transform.rotation.pitch = -90
    spectator.set_transform(transform)
    for waypoint in target_lane_waypoints:
        world.debug.draw_string(waypoint.transform.location + carla.Location(z=0.5), '*', draw_shadow=False,
                                   color=carla.Color(r=255, g=0, b=0), life_time=60.0,
                                   persistent_lines=True)


def generate_candidate_waypoint():
    candidate_waypoints = carla_map.generate_waypoints(300)
    
    selected_waypoints = []
    selected_road = set()
    
    for waypoint in candidate_waypoints:
        if waypoint.lane_type !=  carla.LaneType.Driving and waypoint.road_id in selected_road:
            continue
        
        next_wps = waypoint.next(200)
        if len(next_wps) <= 0:
            continue
        next_wps = random.choice(next_wps)
        if next_wps is not None and next_wps.lane_type == carla.LaneType.Driving:
            selected_waypoints.append(waypoint)
            selected_road.add(waypoint.road_id)
    return selected_waypoints

def generate_route_xml(carla_map, selected_waypoints, weather_dict):
    routes = []
    idx = 0
    grp = GlobalRoutePlanner(carla_map, 1.0)

    for idx, wp in enumerate(selected_waypoints):
        curr_route_config = deepcopy(route_config)
        route_starting_point = wp
        
        curr_route_config["@id"] = idx
        curr_route_config["@town"] = carla_map.name.split("/")[-1]
        weather_idx = random.choice([0, 1, 6, -1])
        curr_route_config["weathers"]["weather"] = weather_dict["weathers"]["case"][weather_idx]["weather"]

        distance = 100
        step_size = 5.0
        step = int(distance/step_size)
        next_wp = route_starting_point
        
        skip = False
        wps = [next_wp]
        for i in range(step):
            loc = next_wp.transform.location
            if len(next_wp.next(step_size)) <= 0:
                skip = True
                break
            next_wp = next_wp.next(step_size)[0]
            wps.append(next_wp)
        if skip:
            continue
        
        if not path_exist(grp, wps):
            continue
        
        for wp in wps:
            loc = wp.transform.location
            curr_route_config["waypoints"]["position"].append({"@x": f"{loc.x:.1f}", "@y": f"{loc.y:.1f}", "@z": f"{loc.z:.1f}"})
        
        trigger_wp = route_starting_point.next(2.0)[0]
        
        curr_route_config['scenarios'][0]['scenario']["@name"] = f"NatualPedestrian_{idx}"
        loc = trigger_wp.transform.location
        rot = trigger_wp.transform.rotation
        curr_route_config['scenarios'][0]['scenario']["trigger_point"] = {"@x": f"{loc.x:.1f}", "@y": f"{loc.y:.1f}", "@z": f"{loc.z:.1f}", "@yaw": f"{rot.yaw:.1f}"}
        routes.append(curr_route_config)
        idx += 1

    xml_string = xmltodict.unparse({"routes" : {"route" : routes}}, pretty=True)

    return xml_string


    
if __name__ == "__main__":
    work_dir = os.path.dirname(os.path.abspath(__file__)) + "/../.."
    weather_xml = f"{work_dir}/Bench2Drive/leaderboard/data/weather.xml"
    save_path = f"{work_dir}/leaderboard_autopilot/data/drivIng/drivIng_long.xml"
    
    client = carla.Client('localhost', 12345) 
    # world = client.load_world('Grid0828')
    world = client.get_world()
    carla_map = world.get_map()
    
    print("generating waypoints...")
    candidate_waypoints = generate_candidate_waypoint()
    weather_dict = None
    with open(weather_xml, "r") as f:
        xml_content = f.read()
        weather_dict = xmltodict.parse(xml_content)
    import pprint; pprint.pp(len(weather_dict["weathers"]["case"]))
    xml_string = generate_route_xml(carla_map, candidate_waypoints, weather_dict)
    
    with open(save_path, "w") as f:
        f.write(xml_string)
    
    print(f"Route XML saved to {save_path}")
