#!/bin/bash

# --- USER CONFIGURATION ---
# Path definitions based on your local workspace structure
PROJECT_ROOT="/home/fquareng/work/ExtremePrecipSR"

# CHANGE THIS: Point to the new DDPM evaluation script
EVAL_SCRIPT_PATH="${PROJECT_ROOT}/src/eval_ddpm.py"

# DEFINE THIS PATH CAREFULLY:
# Where are the experiment folders located?
RUNS_BASE_DIR="/home/fquareng/work/sr_experiment_runs"

# Environment name from your local config
ENV_NAME="dl"

# --- EXPERIMENT DEFINITIONS ---
# PUT YOUR DDPM RUN FOLDER NAME HERE
# Example: "DDPM_SR_20251210_120000"
MODELS_TO_EVAL=("DDPM_SR_20251201_111432")

N_EXPERIMENTS=${#MODELS_TO_EVAL[@]}

# --- ENVIRONMENT SETUP ---
# Sourcing bashrc to ensure conda/mamba is available
source /home/fquareng/.bashrc

echo "----------------------------------------------------"
echo "Machine: $(hostname)"
echo "Environment: $ENV_NAME"
echo "Found ${N_EXPERIMENTS} DDPM experiments to evaluate."
echo "Scripts location: ${EVAL_SCRIPT_PATH}"
echo "Runs location: ${RUNS_BASE_DIR}"
echo "----------------------------------------------------"

# --- EVALUATION LOOP ---
for (( i=0; i<${N_EXPERIMENTS}; i++ )); do
    e=${MODELS_TO_EVAL[i]}
    
    # Construct the full path for the local machine
    CURRENT_RUN_DIR="${RUNS_BASE_DIR}/${e}"

    echo "--- Starting DDPM evaluation for: $e ---"
    
    # Sanity check: verify the data exists locally before attempting to run Python
    if [ ! -d "$CURRENT_RUN_DIR" ]; then
        echo "ERROR: Run directory not found at $CURRENT_RUN_DIR"
        echo "Skipping this experiment..."
        echo "----------------------------------------------------"
        continue
    fi

    # Execute the python script using the local environment
    micromamba run -n "$ENV_NAME" python "$EVAL_SCRIPT_PATH" \
        --run_dir "$CURRENT_RUN_DIR"

    echo "--- Finished evaluation for: $e ---"
    echo ""
done

echo "All DDPM evaluations complete."