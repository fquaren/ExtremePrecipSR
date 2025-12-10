#!/bin/bash

# --- USER CONFIGURATION ---
# Path definitions based on your local workspace structure
PROJECT_ROOT="/home/fquareng/work/ExtremePrecipSR"
EVAL_SCRIPT_PATH="${PROJECT_ROOT}/src/eval_gamma.py"

# DEFINE THIS PATH CAREFULLY:
# Where are the experiment folders (e.g., GammaEmulatorv2_...) located on this machine?
# I have assumed they are in 'experiment_runs' inside your project root.
RUNS_BASE_DIR="/home/fquareng/work/experiment_runs"

# Environment name from your local config
ENV_NAME="dl"

# --- EXPERIMENT DEFINITIONS ---
# Exact arrays from the SLURM script
EMULATORS_TO_EVAL=("GammaEmulator_OfflineMixup_none_Vanilla_2025-12-06_22-35-45")
EMULATORS_TYPE=("soft")
ARCHITECTURE_TYPE=("Vanilla")

N_EXPERIMENTS=${#EMULATORS_TO_EVAL[@]}

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
    e=${EMULATORS_TO_EVAL[i]}
    m=${EMULATORS_TYPE[i]}
    
    # Construct the full path for the local machine
    CURRENT_RUN_DIR="${RUNS_BASE_DIR}/${e}"

    echo "--- Starting evaluation for: $e ---"
    echo "Constraint Mode: $m"
    
    # Sanity check: verify the data exists locally before attempting to run Python
    if [ ! -d "$CURRENT_RUN_DIR" ]; then
        echo "ERROR: Run directory not found at $CURRENT_RUN_DIR"
        echo "Skipping this experiment..."
        echo "----------------------------------------------------"
        continue
    fi

    # Execute the python script using the local environment
    micromamba run -n "$ENV_NAME" python "$EVAL_SCRIPT_PATH" \
        --run_dir "$CURRENT_RUN_DIR" \
        --constraint_mode "$m" \
        --architecture_type "${ARCHITECTURE_TYPE[i]}"

    echo "--- Finished evaluation for: $e ---"
    echo ""
done

echo "All evaluations complete."