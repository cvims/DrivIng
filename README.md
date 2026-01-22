# DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration

<p align="center">
  <a href="https://arxiv.org/abs/2601.15260" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv" alt="arXiv">
  </a>
  <a href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VBZKDY" target="_blank">
    <img src="https://img.shields.io/badge/Dataset-Download-green?logo=databricks" alt="Dataset Download">
  </a>
</p>

---


![DrivIng overview](assets/teaser.jpg)

---
## 1️⃣ Accessing the dataset

### 📦 Downloading and unzipping the dataset
The dataset can be downloaded from [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VBZKDY) or even easier with the provided *download_dataset.py* and afterwards with *unzip_dataset.py* provided in *dataset_scripts/driving_dataset_scripts/* (see conda setup in the next sections).

```bash
conda activate driving_dataset_scripts
python download_dataset.py /path/to/DrivIng_zipped --full
python extract_dataset.py /path/to/DrivIng_zipped /path/to/DrivIng [--delete-chunks] [--delete-tar]
```

### 🗂️ Dataset structure
After the dataset is downloaded and unziped, make sure that it matches the following format.
```yaml
. [DATA_ROOT] # Dataset root folder
├── 📂DrivIng # data files
│   ├── 📂day # day sequence data
│   │   ├── 🏷️annotations.json # All annotations of the sequence (10 Hz)
│   │   ├── 📂middle_lidar # lidar (10 Hz)
│   │   │   ├── 🌫️1750166025000032000.npz # point cloud data
│   │   │   └   ...
│   │   ├── 📂vehicle_back_left_camera # camera (10 Hz)
│   │   │   ├── 🖼️1750166025025979996.jpg # image data
│   │   │   └   ...
│   │   ├── 📂vehicle_back_right_camera
│   │   ├── 📂vehicle_front_left_camera
│   │   ├── 📂vehicle_front_right_camera
│   │   ├── 📂vehicle_left_camera
│   │   ├── 📂vehicle_right_camera
│   │   ├── 📂vehicle_state
│   │   │   ├── 🚘1750166025059999942.json # state information of the vehicle
│   │   │   └   ...
│   │   ├── 🧭calibration.json # all intrinsic and extrinsic calibration parameters
│   │   ├── 📊timesync_info.csv # time synchronization information linking all sensor data together (10 Hz)
│   │   └──📂sweeps
│   │       ├── middle_lidar
│   │       │  ├── 🌫️1750166024950024000.npz # intermediate point clouds in 10 Hz (fused with timesync data it becomes the original 20 Hz)
│   │       │  └   ...
│   │       └── vehicle_state
│   │          ├── 🚘1750166025049999952.json # vehicle state in 100 Hz (fused with timesync data it becomes the original 100 Hz)
│   │          └   ...
│   ├── 📂dusk # dusk sequence data
│   └── 📂night # night sequence data
└── 📂digital_twin # carla digital twin folder
```

---

## 2️⃣ Create environments
For simplicity, we recommend using three separate environment and therefore create 3 different conda environments for the three subfolders *dataset_scripts*, *CARLA_scripts*, and *mmdetection3d*.

### 🧬 Clone the repository
```bash
git clone <TODO>
cd DrivIng
```

### 🧾 Create dataset_scripts environment
Navigate to DrivIng/dataset_scripts.
```bash
conda create --name driving_dataset_scripts python==3.10.18 -y
conda activate driving_dataset_scripts
pip install -r requirements.txt
pip install -e .
```

### 🧾 Create CARLA_scripts environment
Navigate to DrivIng/CARLA_scripts.
```bash
conda deactivate
conda env create -f environment.yml
conda activate carla_scripts
```

### 🧾 Create mmdetection3d environment
Navigate to DrivIng/mmdetection3d. If further instructions are needed, we refer to the official [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) git repository.
We used CUDA 11.7 for all our experiments as well as the package versions as listed below.
```bash
conda deactivate
conda create --name driving_mmdetection3d python==3.9 -y
conda activate driving_mmdetection3d
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -U openmim==0.3.9
mim install mmengine==0.10.7
mim install mmcv==2.0.0.rc4
mim install mmdet==3.3.0
pip install . --no-build-isolation
```

## 3️⃣ Perception models - benchmark evaluation
We use our [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) adaptation to evaluate our DrivIng dataset on different pre-implemented models.

### 🔁 mmdetection3d format conversion
Please download the dataset and unzip by following the above description.
We start with converting the dataset to nuScenes format using the *driving_scripts* environment.
Please change the paths to the correct root and destination directories.
The following scripts will create the sequence splits as well as the file preparation for the nuScenes format.

In case you want to visualize the dataset in different ways, check out the *create_video.py* script in "<git-repo>/dataset_scripts/driving_dataset_scripts". Example usage:
```bash
conda activate driving_scripts
python 
```

Navigate to "<git-repo>/dataset_scripts/driving_dataset_scripts/data_conversion" and produce the dataset splits as follows:
```bash
conda activate driving_scripts
python create_sequences_from_annotation_file.py --data-path <path_to_dataset> --sequence-name [day|dusk|night] [--out-path] [--n-chunks] [--seed]
```
By default *create_sequences_from_annotation_file.py* creates a directory **splits** at the directory level of **--data-path** argument.

For nuScenes format conversion stay in to "<git-repo>/dataset_scripts/driving_dataset_scripts/data_conversion" and execute the next script as follows:
```bash
python create_nuscenes_format.py --split-path <path_to_dataset/splits> --sequence-name [day|dusk|night] --use-multiprocessing [--target-nuscenes-folder]
```
By default *create_nuscenes_format.py* creates a directory **nuScenes_DrivIng** at the directory level of **--split-path** argument.


After the dataset format conversion change the conda environments to create the mmdetection3d format.
```bash
conda deactivate
conda activate driving_mmdetection3d
ln -s <path_to_Driving_nuScenes_format> <path_to_git_repo>/mmdetection3d/data/nuscenes-driving
python tools/create_data.py nuscenes-driving --root-path ./data/nuscenes-driving --out-dir ./data/nuscenes-driving --subfolder [day|dusk|night] 
```


### 🏋️ Training mmdetection3d
Navigate to DrivIng/mmdetection3d*. Example on CenterPoint and day split.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh \
configs_driving/CenterPoint/day/day_centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-3d.py 8 \
--work-dir work_dirs/lidar-only/centerpoint/train_day
```
Use the amount of available graphic cards in CUDA_VISIBLE_DEVICES and additionally correct the number of graphic cards after the *config* argument.
Update the evaluation output directory *--work-dir* based on your needs.

### 🔍 Inference mmdetection3d
Navigate to *DrivIng/mmdetection3d*. Example on CenterPoint and day split. Make sure to have a weights in the *work-dir* argument.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh \
configs_driving/CenterPoint/day/test_eval_day_centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-3d.py \
work_dirs/lidar-only/centerpoint/train_day/best_NuScenesDrivIng_metric_pred_instances_3d_NuScenes_DrivIng_NDS_epoch_[XX].pth 8 \
--work-dir work_dirs/lidar-only/centerpoint/train_day/test_day
```
Use the amount of available graphic cards in CUDA_VISIBLE_DEVICES and additionally correct the number of graphic cards after the *config* argument.
Update the *model weights (pth)* and evaluation output directory *--work-dir* based on your needs.


## 4️⃣ CARLA map integration
Copy both tar.gz files into your CARLA UE4 installation (version: 0.9.15.2) *Import* folder and execute:
```bash
sh ImportAssets.sh
```
This will automatically include the digital twin of DrivIng into your CARLA environment.
Please check out [CARLA scripts README](CARLA_scripts/README.md) to learn more about using our digital twin with our provided scripts.


## 📝 License
- **Code**: Licensed under the **MIT** License. See [LICENSE](LICENSE) file
 for details.

- **Dataset**: Licensed under the Creative Commons Attribution 4.0 International [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en). You must give appropriate credit; Cannot be used for commercial purposes; You may not distribute modified versions of the dataset.
--- 

## 🏆 Acknowledgments
- This work was supported by [AImotion  Bavaria](https://www.thi.de/forschung/aimotion/aboutaimotion/), the [**Hightech Agenda Bavaria**](https://www.hightechagenda.de/), the [**Bavarian Academic Forum - BayWISS**](https://mobilitaet-verkehr.baywiss.de/), all funded by the [_Bavarian State Ministry of Science and the Arts (Bayrisches Staatsministerium für Wissenschaft und Kunst_)](https://www.stmwk.bayern.de/index.html), and by the [**iEXODDUS**](https://iexoddus-project.eu/) project (Grant Agreement No. 101146091).

## 📖 Citation
If you use DrivIng in your research, please cite:
```bibtex
@misc{roessle2026drivinglargescalemultimodaldriving,
      title={DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration}, 
      author={Dominik Rößle and Xujun Xie and Adithya Mohan and Venkatesh Thirugnana Sambandham and Daniel Cremers and Torsten Schön},
      year={2026},
      eprint={2601.15260},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.15260}, 
}
```
