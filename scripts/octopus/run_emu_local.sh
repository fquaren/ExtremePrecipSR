#!/bin/bash

# --- USER CONFIGURATION ---
# Adjusted paths based on your previous local configuration
PROJECT_ROOT="/home/fquareng/work/ExtremePrecipSR"
TRAIN_SCRIPT_PATH="${PROJECT_ROOT}/src/train_gamma.py"
DATA_SCRIPT_PATH="${PROJECT_ROOT}/data/mixup_dataset.py"
LOG_DIR="${PROJECT_ROOT}/logs/GammaEmulators_$(date +%Y%m%d_%H%M%S)"
ENV_NAME="dl"

# Create log directory to ensure reproducibility
mkdir -p "$LOG_DIR"

# --- ENVIRONMENT SETUP ---
source /home/fquareng/.bashrc

echo "----------------------------------------------------"
echo "Machine: $(hostname)"
echo "Logs will be saved to: $LOG_DIR"
echo "Environment: $ENV_NAME"
echo "----------------------------------------------------"

# --- 1. DATA GENERATION STEP ---
echo "--- Running Offline Data Generation (MixUp) ---"
DATA_LOG="${LOG_DIR}/data_generation.log"

# Run the data generation script and log output
micromamba run -n "$ENV_NAME" python "$DATA_SCRIPT_PATH" > "$DATA_LOG" 2>&1

if [ $? -eq 0 ]; then
    echo "Data generation successful."
else
    echo "CRITICAL FAILURE: Data generation failed. Check $DATA_LOG"
    exit 1
fi
echo "----------------------------------------------------"

# --- 2. DEFINE EXPERIMENT CONFIGURATIONS ---
# Parallel arrays to define specific pairs of (Architecture, Constraint Mode)
# Matches the logic of the SLURM script exactly.
ARCHS=("Attention" "Attention" "Attention" "Vanilla")
MODES=("hard" "soft" "none" "none")

N_EXPS=${#ARCHS[@]}

echo "Found ${N_EXPS} specific experiments to train."
echo "----------------------------------------------------"

# --- 3. TRAINING LOOP ---
for (( i=0; i<${N_EXPS}; i++ )); do
    curr_arch=${ARCHS[i]}
    curr_mode=${MODES[i]}
    
    LOG_FILE="${LOG_DIR}/train_${curr_mode}_${curr_arch}.log"
    
    echo "Starting Experiment $((i+1))/${N_EXPS}: Mode=$curr_mode | Arch=$curr_arch"
    echo "Logging to: $LOG_FILE"

    # Run training
    micromamba run -n "$ENV_NAME" python "$TRAIN_SCRIPT_PATH" \
        --constraint_mode "$curr_mode" \
        --arch "$curr_arch" \
        > "$LOG_FILE" 2>&1

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "SUCCESS: $curr_mode, $curr_arch"
    else
        echo "FAILURE: $curr_mode, $curr_arch - Check log for details."
    fi
    echo "----------------------------------------------------"
done

echo "All training experiments complete."