#!/bin/bash

# --- USER CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ExtremePrecipSR"
EVAL_SCRIPT_PATH="${PROJECT_ROOT}/src/eval_gamma.py"
RUNS_BASE_DIR="/home/fquareng/work/experiment_runs"
ENV_NAME="dl"

# --- EXPERIMENT DEFINITIONS ---
EMULATORS_TO_EVAL=("GammaEmulator_v6_Constrained_SingleRun_2026-01-16_10-04-48" "GammaEmulator_v6_Isometric_SingleRun_2026-01-16_10-04-59")
ARCHS=("Constrained" "Isometric")

N_EXPERIMENTS=${#EMULATORS_TO_EVAL[@]}

# --- ENVIRONMENT SETUP ---
source /home/fquareng/.bashrc

echo "----------------------------------------------------"
echo "Machine: $(hostname)"
echo "Environment: $ENV_NAME"
echo "Found ${N_EXPERIMENTS} experiments to evaluate."
echo "----------------------------------------------------"

# --- EVALUATION LOOP ---
for (( i=0; i<${N_EXPERIMENTS}; i++ )); do
    # Capture variables for this iteration
    e=${EMULATORS_TO_EVAL[i]}
    a=${ARCHS[i]}
    CURRENT_RUN_DIR="${RUNS_BASE_DIR}/${e}"

    # START PARALLEL BLOCK
    # We open a subshell ( ... ) and background it with &
    (
        echo "[Job $i] Checking: $e ($a)"
        
        # Sanity check
        if [ ! -d "$CURRENT_RUN_DIR" ]; then
            echo "[Job $i] ERROR: Run directory not found: $CURRENT_RUN_DIR"
            exit 1
        fi

        # Define a specific log file to avoid console clutter
        LOG_FILE="${CURRENT_RUN_DIR}/eval_${a}.log"
        echo "[Job $i] Starting Python script. Logs -> $LOG_FILE"

        # Execute python
        # We redirect both stdout (1) and stderr (2) to the log file
        micromamba run -n "$ENV_NAME" python "$EVAL_SCRIPT_PATH" \
            --run_dir "$CURRENT_RUN_DIR" \
            --arch "$a" > "$LOG_FILE" 2>&1

        echo "[Job $i] Finished: $e"
    ) & 
    # END PARALLEL BLOCK (The loop continues immediately to the next item)

done

echo "Jobs launched. Waiting for all evaluations to complete..."
wait
echo "All evaluations complete."