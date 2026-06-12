#!/usr/bin/env bash
set -euo pipefail

POSEINSERT_ROOT="${POSEINSERT_ROOT:-/home/user/wyw/PoseInsert}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/user/wyw/RoboTwin}"
CONDA_BIN="${CONDA_BIN:-/home/user/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-RoboTwin}"

TASK_NAME="handover_block"
TASK_CONFIG="handover_block_pose_clean50"
SOURCE_DIR="/data1/RoboTwin_data/RoboTwin_Clean_Data/dataset/handover_block/aloha-agilex_clean_50"
SAVE_BASE="${POSEINSERT_ROOT}/pose_object_data"
TARGET_DIR="${SAVE_BASE}/${TASK_NAME}/${TASK_CONFIG}"
CONFIG_PATH="${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml"

echo "POSEINSERT_ROOT=${POSEINSERT_ROOT}"
echo "ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
echo "SOURCE_DIR=${SOURCE_DIR}"
echo "TARGET_DIR=${TARGET_DIR}"
echo "CONFIG_PATH=${CONFIG_PATH}"

test -d "${POSEINSERT_ROOT}"
test -d "${ROBOTWIN_ROOT}"
test -f "${SOURCE_DIR}/seed.txt"
test -d "${SOURCE_DIR}/_traj_data"

cd "${ROBOTWIN_ROOT}"
git checkout main
echo "RoboTwin branch: $(git branch --show-current)"
git status --short

if ! grep -q "object_pose" "${ROBOTWIN_ROOT}/envs/handover_block.py"; then
  echo "ERROR: ${ROBOTWIN_ROOT}/envs/handover_block.py does not appear to save object_pose." >&2
  exit 2
fi

mkdir -p "${TARGET_DIR}"
cp -n "${SOURCE_DIR}/seed.txt" "${TARGET_DIR}/seed.txt"
mkdir -p "${TARGET_DIR}/_traj_data"
cp -an "${SOURCE_DIR}/_traj_data/." "${TARGET_DIR}/_traj_data/"

cat > "${CONFIG_PATH}" <<YAML
# Recollect official handover_block clean_50 seeds with per-frame object_pose.
render_freq: 0
episode_num: 50
use_seed: true
save_freq: 15
embodiment: [aloha-agilex]
language_num: 4
domain_randomization:
  random_background: false
  cluttered_table: false
  clean_background_rate: 1
  random_head_camera_dis: 0
  random_table_height: 0
  random_light: false
  crazy_random_light_rate: 0
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  third_view: false
  depth: false
  pointcloud: false
  observer: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
pcd_down_sample_num: 1024
pcd_crop: true
save_path: ${SAVE_BASE}
clear_cache_freq: 5
collect_data: true
eval_video_log: false
YAML

echo "Prepared seed count: $(wc -w < "${TARGET_DIR}/seed.txt")"
echo "Prepared traj count: $(find "${TARGET_DIR}/_traj_data" -maxdepth 1 -name 'episode*.pkl' | wc -l)"

"${CONDA_BIN}" run -n "${CONDA_ENV}" python script/collect_data.py "${TASK_NAME}" "${TASK_CONFIG}"
