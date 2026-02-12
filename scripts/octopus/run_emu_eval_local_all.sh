#!/bin/bash

# --- USER CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ExtremePrecipSR"
EVAL_SCRIPT_PATH="${PROJECT_ROOT}/src/eval_gamma.py"
RUNS_BASE_DIR="/home/fquareng/work/final_experiment_runs"
ENV_NAME="dl"

# CONCURRENCY LIMIT
# Adjust this based on your GPU VRAM and System RAM. 
# e.g., if one run takes 8GB VRAM and you have 24GB, set to 2 or 3.
MAX_JOBS=4 

# --- ENVIRONMENT SETUP ---
source /home/fquareng/.bashrc

if [ ! -d "$RUNS_BASE_DIR" ]; then
    echo "CRITICAL ERROR: Run directory $RUNS_BASE_DIR does not exist."
    exit 1
fi

echo "----------------------------------------------------"
echo "Machine: $(hostname)"
echo "Environment: $ENV_NAME"
echo "Concurrency Limit: $MAX_JOBS jobs"
echo "Scanning for final experiments in: $RUNS_BASE_DIR"
echo "----------------------------------------------------"

# --- EVALUATION LOOP ---
for run_path in "${RUNS_BASE_DIR}"/GammaEmulator_v6_*; do
    
    # 1. Validate directory
    if [ ! -d "$run_path" ]; then
        continue
    fi

    # 2. Parse Architecture
    e=$(basename "$run_path")
    
    # Logic: Remove prefix "GammaEmulator_v6_" then remove suffix starting at "_SingleRun"
    temp_str="${e#GammaEmulator_v6_}"
    a="${temp_str%%_SingleRun*}"

    if [ -z "$a" ]; then
        echo "[Skipping] Could not detect architecture in: $e"
        continue
    fi

    CURRENT_RUN_DIR="$run_path"

    # --- SEMAPHORE LOGIC ---
    # Check number of background jobs. If >= MAX_JOBS, sleep until one finishes.
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 5
    done

    # --- JOB LAUNCH ---
    (
        echo "[Started] $e (Arch: $a)"
        
        LOG_FILE="${CURRENT_RUN_DIR}/eval_${a}.log"

        micromamba run -n "$ENV_NAME" python "$EVAL_SCRIPT_PATH" \
            --run_dir "$CURRENT_RUN_DIR" \
            --arch "$a" > "$LOG_FILE" 2>&1

        echo "[Finished] $e"
    ) & 

done

echo "All jobs queued. Waiting for remaining active jobs to complete..."
wait
echo "All evaluations complete."