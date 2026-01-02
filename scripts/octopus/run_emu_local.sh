#!/bin/bash

# ==============================================================================
# SCRIPT: run_emulator_ablation.sh
# DESCRIPTION: Runs the 3-stage ablation study for the Gamma Emulator v4.
#              1. Attention (Gated CNN, no constraints)
#              2. Attention + Soft Constraints
#              3. Attention + Hard Constraints
# ==============================================================================

# --- USER CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ExtremePrecipSR"
TRAIN_SCRIPT_PATH="${PROJECT_ROOT}/src/train_gamma_v4.py"
DATA_SCRIPT_PATH="${PROJECT_ROOT}/data/mixup_dataset.py"
LOG_DIR="${PROJECT_ROOT}/logs/GammaEmulators_Ablation_$(date +%Y%m%d_%H%M%S)"
ENV_NAME="dl"

# Create log directory
mkdir -p "$LOG_DIR"

# --- ENVIRONMENT SETUP ---
# Ensure micromamba/conda is available
source /home/fquareng/.bashrc

echo "======================================================================"
echo "STARTING EMULATOR ABLATION STUDY"
echo "Machine: $(hostname)"
echo "Logs Directory: $LOG_DIR"
echo "Environment: $ENV_NAME"
echo "======================================================================"

# --- STEP 1: DATA GENERATION CHECK ---
# We check if data exists; if not, we run the generation script.
echo ">>> [Step 1/2] Checking Data Availability..."

DATA_LOG="${LOG_DIR}/data_generation.log"
MIXUP_FILE="/home/fquareng/work/data/extremes/OPERA/patches/precip/train/mixup_augmented_precip.npz"

# if [ -f "$MIXUP_FILE" ]; then
    # echo "MixUp data found at $MIXUP_FILE. Skipping generation."
# else
    # echo "MixUp data NOT found. Running generation script..."
micromamba run -n "$ENV_NAME" python "$DATA_SCRIPT_PATH" > "$DATA_LOG" 2>&1
    
    # if [ $? -eq 0 ]; then
        # echo "Data generation successful."
    # else
        # echo "CRITICAL FAILURE: Data generation failed. See $DATA_LOG"
        # exit 1
    # fi
# fi
echo "----------------------------------------------------------------------"

# --- STEP 2: RUNNING EXPERIMENTS ---
echo ">>> [Step 2/2] Launching Training Loops..."

# Define the 3 Experiments
# Arrays must be same length. Index i in ARCHS corresponds to Index i in MODES.

# Order:
# 1. Gated CNN (Attention, no constraints)
# 2. Gated + Soft (Attention, soft constraints)
# 3. Gated + Hard (Attention, hard constraints)

ARCHS=("Attention" "Attention" "Attention")
MODES=("none"      "soft"      "hard")

N_EXPS=${#ARCHS[@]}

for (( i=0; i<${N_EXPS}; i++ )); do
    curr_arch=${ARCHS[i]}
    curr_mode=${MODES[i]}
    
    # Create a descriptive log name
    EXP_ID=$((i+1))
    LOG_FILE="${LOG_DIR}/exp${EXP_ID}_${curr_arch}_${curr_mode}.log"
    
    echo "----------------------------------------------------------------------"
    echo "Running Experiment ${EXP_ID}/${N_EXPS}"
    echo "   Architecture:    $curr_arch"
    echo "   Constraint Mode: $curr_mode"
    echo "   Log File:        $LOG_FILE"
    
    # Run Python Script
    micromamba run -n "$ENV_NAME" python "$TRAIN_SCRIPT_PATH" \
        --constraint_mode "$curr_mode" \
        --arch "$curr_arch" \
        --data_percentage 10 \
        > "$LOG_FILE" 2>&1

    # Check Exit Status
    if [ $? -eq 0 ]; then
        echo "   STATUS: SUCCESS"
    else
        echo "   STATUS: FAILED (Check log for details)"
        # Optional: exit 1 # Uncomment to stop pipeline on first failure
    fi
done

echo "======================================================================"
echo "Ablation Study Complete."
echo "======================================================================"