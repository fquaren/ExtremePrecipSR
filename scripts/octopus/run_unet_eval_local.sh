#!/bin/bash

# --- USER CONFIGURATION ---
# Path definitions based on your local workspace structure
PROJECT_ROOT="/home/fquareng/work/ExtremePrecipSR"
EVAL_SCRIPT_PATH="${PROJECT_ROOT}/src/eval_sr.py"

# DEFINE THIS PATH CAREFULLY:
# Where are the experiment folders (e.g., GammaEmulatorv2_...) located on this machine?
# I have assumed they are in 'experiment_runs' inside your project root.
RUNS_BASE_DIR="/home/fquareng/work/sr_experiment_runs"

# Environment name from your local config
ENV_NAME="dl"

# --- EXPERIMENT DEFINITIONS ---
# Exact arrays from the SLURM script
MODELS_TO_EVAL=("GammaEmulatorv3_train_2025-12-11_00-25-05")

N_EXPERIMENTS=${#MODELS_TO_EVAL[@]}

# --- ENVIRONMENT SETUP ---
# Sourcing bashrc to ensure conda/mamba is available
source /home/fquareng/.bashrc

echo "----------------------------------------------------"
echo "Machine: $(hostname)"
echo "Environment: $ENV_NAME"
echo "Found ${N_EXPERIMENTS} experiments to evaluate."
echo "Scripts location: ${EVAL_SCRIPT_PATH}"
echo "Runs location: ${RUNS_BASE_DIR}"
echo "----------------------------------------------------"

# --- EVALUATION LOOP ---
for (( i=0; i<${N_EXPERIMENTS}; i++ )); do
    e=${MODELS_TO_EVAL[i]}
    
    # Construct the full path for the local machine
    CURRENT_RUN_DIR="${RUNS_BASE_DIR}/${e}"

    echo "--- Starting evaluation for: $e ---"
    
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

echo "All evaluations complete."