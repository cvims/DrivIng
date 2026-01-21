import os
import json
import argparse

def load_data_dirs(data_dir):
    """
    Loads the directories containing scenario data for each route.

    Args:
        data_dir (str): The root directory containing route subdirectories.

    Returns:
        dict: A dictionary where keys are route names and values are sets of
              absolute paths to scenario directories within that route.
              Returns an empty dictionary if data_dir does not exist or is empty.
    """
    data_dirs = {}
    for routes in os.listdir(data_dir):

        routes_dir = os.path.join(data_dir, routes)
        if os.path.isdir(routes_dir):
            scenario_dirs = [
                os.path.join(routes_dir, scenario)
                for scenario in os.listdir(routes_dir)
                if os.path.isdir(os.path.join(routes_dir, scenario))
            ]
            data_dirs[routes] = set(scenario_dirs)  # Use set for fast removal
    return data_dirs


def check_route(result_dir, data_dirs):
    """
    Identifies and removes scenario directories from `data_dirs` that correspond to failed routes
    based on the information in the result files.

    Args:
        result_dir (str): The directory containing JSON result files.
        data_dirs (dict): A dictionary (modified in-place) where keys are route names and
                           values are sets of scenario directory paths.
    """
    for file in os.listdir(result_dir):
        # only process json files
        # the json files contain the status of each route
        # if the route failed, remove the corresponding scenario directory from `data_dirs`
        if not file.endswith(".json"):
            continue
        file_prefix = file[:-7]
        file_path = os.path.join(result_dir, file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            for route in data['_checkpoint']["records"]:
                data_prefix = f"{file[:-5]}_expert_traj_route{route['index']}"
                if ("TickRuntime" in route['status'] or "Simulation crashed" in route['status']) and "deviated " not in route['status']:
                    print(f"{route['save_name']} - {route['status']}")
                    # Remove matching dirs in-place
                    to_remove = {rd for rd in data_dirs.get(file_prefix, set()) if data_prefix in rd}
                    for rd in to_remove:
                        data_dirs[file_prefix].remove(rd)


def generate_train_val_split(data_dirs, save_path, val_routes):
    """
    Generates a JSON file defining the train/validation split based on the provided data directories
    and validation routes.

    Args:
        data_dirs (dict): A dictionary where keys are route names and values are sets of scenario directory paths.
        save_path (str): The path to save the generated JSON file.
        val_routes (set): A set of route names to be used for validation.  All other routes are used for training.
    """
    json_dict = {'train': [], 'val': []}
    for _, value in data_dirs.items():
        for scenario_dir in value:
            route_id = int(scenario_dir.split('_')[2])
            if route_id in val_routes:
                json_dict['val'].append(scenario_dir)
            else:
                json_dict['train'].append(scenario_dir)
    with open(save_path, 'w') as f:
        json.dump(json_dict, f, indent=4)


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default="/home/xujun/DrivIng_carla",
                       help='Root directory of the dataset')
    parser.add_argument("--split_name", type=str, default="v1_train_val.json",)
    args = parser.parse_args()
    
    dataset_root = args.dataset_root
    result_dir = f"{dataset_root}/results/expert_traj"
    data_dir = f"{dataset_root}/data"
    save_path = f"{dataset_root}/splits/{args.split_name}"
    val_routes = {0, 1, 2, 3}  # Example validation routes
    

    data_dirs = load_data_dirs(data_dir)
    check_route(result_dir, data_dirs)
    generate_train_val_split(data_dirs, save_path, val_routes)