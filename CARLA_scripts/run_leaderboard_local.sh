# 10.116.88.40
export CARLA_ROOT=<your carla workspace path>

export WORK_DIR="$(pwd -P)"
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner_autopilot
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard_autopilot
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}"::${PYTHONPATH}

export ROUTES_NAME=demo
export ROUTES=${WORK_DIR}/leaderboard_autopilot/data/drivIng/${ROUTES_NAME}.xml
export SAVE_DIR=<your save path>

export TEAM_AGENT=${WORK_DIR}/team_code/driving_datagen.py
export TEAM_CONFIG=${WORK_DIR}/team_code/config.py

export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export RESUME=0
export SEED=42
export CHECKPOINT_ENDPOINT=${SAVE_DIR}/results/${ROUTES_NAME}.json
export DEBUG_ENV_AGENT=0
export RECORD=1
export DIRECT=1
export COMPILE=0
export TOWN=eval
export REPETITION=0
export DATAGEN=1
export TUNED_AIM_DISTANCE=0
export SLOWER=0
export UNCERTAINTY_WEIGHT=1
export STOP_AFTER_METER=-1
export SAVE_PATH=${SAVE_DIR}/data
export PORT=12345
export TM_PORT=1234
export DEBUG_CHALLENGE=0

export CUDA_VISIBLE_DEVICES=1

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=0 \
--traffic-manager-seed=${SEED} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--timeout=100 \
--traffic-manager-port=${TM_PORT}