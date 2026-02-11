# CARLA map integration
- [CARLA map integration](#carla-map-integration)
  - [CARLA installation](#carla-installation)
  - [Install DrivIng-CARLA map](#install-driving-carla-map)
  - [Open DrivIng-CARLA map](#open-driving-carla-map)
  - [Kinematic Reply](#kinematic-reply)
  - [Generate auto-generated traffic flow](#generate-auto-generated-traffic-flow)
  - [Pre-process training data with CARLA](#pre-process-training-data-with-carla)
    - [STEP1: Dataset Split Generator](#step1-dataset-split-generator)
    - [STEP2: CARLA Dataset Converter \& Preprocessor](#step2-carla-dataset-converter--preprocessor)
    - [STEP3: NuScenes Format Converter](#step3-nuscenes-format-converter)
    - [STEP1-3: Three Steps All in One](#step1-3-three-steps-all-in-one)
    - [STEP4: Build Dataset](#step4-build-dataset)

## CARLA installation
Download and setup CARLA 0.9.15

```bash
    mkdir carla
    cd carla
    wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    tar -xvf CARLA_0.9.15.tar.gz
    cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
    cd .. && bash ImportAssets.sh
    export CARLA_ROOT=YOUR_CARLA_PATH
    echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/YOUR_CONDA_ENV_NAME/lib/python3.7/site-packages/carla.pth # python 3.8 also works well, please set YOUR_CONDA_PATH and YOUR_CONDA_ENV_NAME
```

## Install DrivIng-CARLA map 
Download and copy both tar.gz files into your CARLA UE4 root folder (version: 0.9.15) *Import* folder
```yaml
. CarlaUE4 # your carla root directory
├── 📂Import 
│   ├── 🌫️ 0821Grid.tar.gz
│   └── 🌫️ 0826Grid.tar.gz
├── ImportAssets.sh
└── ...
```

Then execute 
```bash
sh ImportAssets.sh
```

## Open DrivIng-CARLA map
The name of the carla town is `Grid0828`. You can switch carla town based on the CARLA instruction
[link](https://carla.readthedocs.io/en/latest/tuto_G_getting_started/). This process might take about 20-60 seconds until the spectator get respond.
```python
import carla
import random

client = carla.Client('localhost', 2000)
client.load_world('Grid0828')
```

## Kinematic Reply
This tool replays pre-recorded trajectory datasets (ego vehicle and NPCs) within the CARLA simulator to generate synchronized sensor data. It handles actor spawning, frame-by-frame teleportation, and sensor data collection (RGB, Depth, Semantic).

```bash
cd kinematic_reply
python kinematic_reply.py --dataset_root /path/to/data --output_dir /path/to/output
```

|Argument | Default | Description |
|:--------|:-------:|:------------|
|--dataset_root | [Path] | Root directory containing trajectory data and annotations. |
|--output_dir   | [Path] | Directory where sensor data will be saved. |
|--port         |  2000  | CARLA server TCP port. |
|--start_timestamp |  -1 | The specific timestamp (nanoseconds) to begin the replay. |
|--total_frames |   200  | Number of frames to simulate and record. |
|--save_depth   | False  | Flag to enable depth camera recording. |
|--save_semantic| False  | Flag to enable semantic segmentation recording. |

The script creates a `timestamp` folder inside output_dir containing the generated sensor data for every ticked frame.

## Generate auto-generated traffic flow
The project was build based on CARLA leaderboard 2.0 framework. Data collection agent is implemented based on [CARLA_Garage](https://github.com/autonomousvision/carla_garage)

first modify environment variable in following files
- CARLA_scripts\leaderboard_autopilot\scripts\recode_routes.sh
    - CARLA_ROOT
- CARLA_scripts\run_leaderboard_local.sh
    - CARLA_ROOT
    - SAVE_DIR
- CARLA_scripts\run_multi_leaderboard.sh
    - SAVE_DIR

Generate data with spectator 
```bash
cd <your carla work space>
sh CarlaUE4.sh

cd <DrivIng_root>/CARLA_scripts
bash run_leaderboard_local.sh
```

Generate data without spectator. It executes carla simulator and data generation in different GPUs
```bash
bash run_multi_leaderboard.sh
```

## Pre-process training data with CARLA

### STEP1: Dataset Split Generator

This script scans a dataset of simulation scenarios, filters out failed runs (crashes or runtime errors), and generates a clean train/validation split JSON file.

**Features**

* **Automatic Filtering:** Parses result logs to exclude scenarios marked as "Simulation crashed" or "TickRuntime" errors (unless marked as deviated).
* **Flexible Splitting:** Allows you to specify exactly which route IDs belong to the **validation** set via command line arguments. All other routes are assigned to the **training** set.
* **Output:** Saves a JSON file containing lists of valid directory paths.

**Expected Directory Structure**

The script expects the `dataset_root` to contain the following structure:

```text
dataset_root/
├── data/                  # Subdirectories containing the actual scenario data
├── results/expert_traj/   # JSON files containing simulation status logs
└── splits/                # (Created if missing) Where the output JSON will be saved
```

**Usage**

### STEP2: CARLA Dataset Converter & Preprocessor

This script processes raw data from CARLA simulations (the "DrivIng" dataset) and converts it into a standardized format suitable for training autonomous driving models. It handles coordinate transformations, sensor synchronization, LiDAR conversion, and annotation formatting.

**Features**

* **Parallel Processing:** Uses `pqdm` to process multiple scenes concurrently for high throughput.
* **Coordinate Transformation:** Converts bounding boxes from World coordinates → Ego coordinates → LiDAR/Camera coordinates.
* **Data Optimization:** * Converts `.las` LiDAR files to compressed `.npz` arrays.
  * Uses symbolic links for heavy image data to save disk space.
* **Standardization:** Maps CARLA object names to NuScenes-style categories (e.g., `vehicle.ford.ambulance` → `Car` with attribute `emergency`).
* **filtering:** Automatically filters out objects with fewer than 5 LiDAR points.

**Prerequisites**

Ensure you have the following Python libraries installed:

```bash
pip install numpy pandas laspy tqdm pqdm
```

```bash
cd dataset_scripts/driving_dataset_scripts/data_conversion
sh create_carla_dataset.sh
```

**Directory Structure**
The script expects the input dataset to follow this structure:

```text
dataset_root/
├── data/
│   └── drivIng_long/
│       ├── scene_001/
│       │   ├── anno/       # Gzipped JSON annotations
│       │   ├── lidar/      # .las files
│       │   └── camera/     # RGB images
│       └── ...
├── splits/
│   └── v1_train_val.json   # Split file defining train/val scenes
└── converted/              # (Created automatically) Output directory
```

**Usage**

```bash
python process_data.py --dataset_root /path/to/dataset --num_workers 16
```
**Output Format**

For every processed scene, the script generates:
- **calibration.json**: Intrinsic and extrinsic parameters for cameras and LiDAR.
- **timesync_info.csv**: Maps timestamps to specific sensor file paths.
- **middle_lidar/*.npz**: Point cloud data containing x, y, z, intensity, timestamp.
- **scene_id.json**: The main annotation file containing tracked objects with dimensions, positions, and orientations.
- **Symlinks**: Camera folders (front_left, back_right, etc.) are symlinked to the original source to avoid duplication.

### STEP3: NuScenes Format Converter
This script converts the processed "DrivIng" dataset (or similarly formatted custom datasets) into the standard [NuScenes v1.0 format](https://www.nuscenes.org/nuscenes). It generates the relational database tables (JSON) required to use the dataset with standard autonomous driving toolkits (e.g., mmdetection3d, nuscenes-devkit).

**Features**

* **Parallel Processing:** Utilizes `pqdm` to process sequences concurrently, significantly reducing conversion time.
* **Schema Generation:** Creates all standard NuScenes tables: `scene`, `sample`, `sample_data`, `ego_pose`, `calibrated_sensor`, `sample_annotation`, `instance`, `category`, `attribute`, `visibility`, `sensor`, `log`, and `map`.
* **Coordinate Transformation:** Handles transformations from LiDAR coordinates to Ego vehicle coordinates and generates the necessary Quaternions for NuScenes.
* **Class Mapping:** Automatically maps custom object classes (e.g., "Van", "Cyclist") to NuScenes hierarchies (e.g., `vehicle.car`, `vehicle.bicycle`) and assigns attributes (e.g., `vehicle.parked`, `pedestrian.walking`).
* **Point Filtering:** Calculates the number of LiDAR points inside every ground truth bounding box (`num_lidar_pts`).

**Directory Structure**

The script expects the input data (processed in the previous step) and outputs the NuScenes structure as follows:
```text
dataset_root/
├── converted/                  # Input: Data processed by the previous script
│   ├── sequence_001/
│   │   ├── sequence_001.json
│   │   ├── calibration.json
│   │   ├── timesync_info.csv
│   │   ├── middle_lidar/
│   │   └── ...
│   └── ...
├── splits/
│   └── v1_train_val.json       # Input: Split definition
└── dataset/                    # Output: Generated NuScenes format
    └── v1.0/
        ├── v1.0-trainval/      # Contains scene.json, sample.json, etc.
        └── samples/            # Contains raw sensor data (symlinked or copied)
```

**Usage**
```bash
python convert_to_nuscenes.py --dataset_root /path/to/dataset --split_name v1_train_val.json --num_workers 32
```

### STEP1-3: Three Steps All in One
```bash
bash create_carla_dataset.sh
```
please make sure variables in shell script correct.

### STEP4: Build Dataset

```bash
cd mmdetection3d/tools
ln -s <path_to_Driving_nuScenes_format> <path_to_git_repo>/mmdetection3d/data/nuscenes-driving/carla
python tools/create_data.py nuscenes-driving --root-path ./data/nuscenes-driving --out-dir ./data/nuscenes-driving --subfolder carla
```
