#!/bin/bash
TEAM_AGENT=team_code/driving_datagen.py

CKPT_PATH=
export TEAM_CONFIG=team_code/config.py

PLANNER_TYPE=traj
ALGO=expert

ROOT_ROUTES=leaderboard_autopilot/data/drivIng
SAVE_DIR=<your save path>

BASE_PORT=54000
BASE_TM_PORT=56000

echo -e "**************\033[36m Please Manually adjust GPU or TASK_ID \033[0m **************"
# Example, 8*H100, 1 task per gpu
GPU_RANK_LIST=(0 0 0 0)
ROUTES_LIST=("drivIng_long" "drivIng_long" "drivIng_long" "drivIng_long")
TASK_LIST=(0 1 2 3)   

echo -e "\033[32m GPU_RANK_LIST: $GPU_RANK_LIST \033[0m"
echo -e "\033[32m TASK_LIST: $TASK_LIST \033[0m"
echo -e "***********************************************************************************"

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<$length; i++ )); do
    PORT=$((BASE_PORT + i * 150))
    TM_PORT=$((BASE_TM_PORT + i * 150))
    BASE_ROUTES=${ROOT_ROUTES}/${ROUTES_LIST[$i]}
    SAVE_PATH=${SAVE_DIR}/data/${ROUTES_LIST[$i]}
    CHECKPOINT_PATH=${SAVE_DIR}/results/${ALGO}_${PLANNER_TYPE}
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${CHECKPOINT_PATH}/${ROUTES_LIST[$i]}_${TASK_LIST[$i]}.json"
    GPU_RANK=${GPU_RANK_LIST[$i]}
    echo -e "\033[32m ALGO: $ALGO \033[0m"
    echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
    echo -e "\033[32m TASK_ID: $i \033[0m"
    echo -e "\033[32m PORT: $PORT \033[0m"
    echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
    echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
    echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
    echo -e "\033[32m bash leaderboard/scripts/recode_routes.sh $PORT $TM_PORT $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
    echo -e "***********************************************************************************"
    bash -e leaderboard_autopilot/scripts/recode_routes.sh $PORT $TM_PORT $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log &
    sleep 10
done
wait