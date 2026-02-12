#!/bin/bash

# --- CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ExtremePrecipSR" 
SCRIPT_PATH="${PROJECT_ROOT}/src/train_sr_unet.py"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"

# distinct log files for parallel execution to avoid interleaving
LOG_FILE_NONE="${LOG_DIR}/unet_run_none_p10_${TIMESTAMP}.log"
LOG_FILE_TRAIN="${LOG_DIR}/unet_run_train_p10_${TIMESTAMP}.log"

# --- HARDWARE SETTINGS ---
# Explicitly set the GPU
export CUDA_VISIBLE_DEVICES=0

# Force Python to flush stdout/stderr immediately
export PYTHONUNBUFFERED=1

# --- EXECUTION ---
echo "Starting PARALLEL training on RTX 6000..."
source /home/fquareng/.bashrc

echo "Running final data preprocessing step..."
micromamba run -n dl python "${PROJECT_ROOT}/data/final_preprocessing.py"


# 1. Launch UNet with metric_loss_mode = none (Background process)
echo "Launching Experiment 1: Metric=None, Data=10%"
echo "  -> Logging to: $LOG_FILE_NONE"
micromamba run -n dl python "$SCRIPT_PATH" \
    --metric_loss_mode none \
    --data_percentage 10 \
    > "$LOG_FILE_NONE" 2>&1 &
PID_1=$!

# 2. Launch UNet with metric_loss_mode = train (Background process)
echo "Launching Experiment 2: Metric=Train, Data=10%"
echo "  -> Logging to: $LOG_FILE_TRAIN"
micromamba run -n dl python "$SCRIPT_PATH" \
    --metric_loss_mode train \
    --data_percentage 10 \
    > "$LOG_FILE_TRAIN" 2>&1 &
PID_2=$!

# --- WAIT ---
echo "Both jobs launched (PIDs: $PID_1, $PID_2). Waiting for completion..."
wait

echo "All training jobs finished."