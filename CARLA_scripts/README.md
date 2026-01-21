## CARLA map integration
### CARLA installation
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

### Install DrivIng-CARLA map 
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

### Open DrivIng-CARLA map
The name of the carla town is `Grid0828`. You can switch carla town based on the CARLA instruction
[link](https://carla.readthedocs.io/en/latest/tuto_G_getting_started/). This process might take about 20-60 seconds until the spectator get respond.
```python
import carla
import random

client = carla.Client('localhost', 2000)
client.load_world('Grid0828')
```

### Generate auto-generated traffic flow
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
