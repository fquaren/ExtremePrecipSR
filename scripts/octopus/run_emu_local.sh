#!/bin/bash

# ==============================================================================
# SCRIPT: run_emulator_ablation.sh
# DESCRIPTION: Runs 5 sequential batches. Each batch runs 3 architectures in parallel.
# ==============================================================================

PROJECT_ROOT="/home/fquareng/work/ExtremePrecipSR"
TRAIN_SCRIPT_PATH="${PROJECT_ROOT}/src/train_gamma_v5.py"
LOG_DIR="${PROJECT_ROOT}/logs/GammaEmulators_Ablation_$(date +%Y%m%d_%H%M%S)"
ENV_NAME="dl"

mkdir -p "$LOG_DIR"

# --- ENVIRONMENT SETUP ---
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ENV_NAME"

echo "======================================================================"
echo "STARTING BATCHED ABLATION STUDY"
echo "Strategy: 5 Sequential Batches x 3 Parallel Jobs"
echo "======================================================================"

ARCHS=("Baseline" "Isometric" "Constrained") 
SEEDS=(42 43 44 45 46)

# Counter for visual progress
BATCH_NUM=1
TOTAL_BATCHES=${#SEEDS[@]}

# --- OUTER LOOP: SEQUENTIAL (Iterate through Seeds) ---
for seed in "${SEEDS[@]}"; do
    
    echo "----------------------------------------------------------------------"
    echo "Starting Batch ${BATCH_NUM}/${TOTAL_BATCHES} | Seed: ${seed}"
    echo "----------------------------------------------------------------------"

    # --- INNER LOOP: PARALLEL (Iterate through Architectures) ---
    for arch in "${ARCHS[@]}"; do
        
        # CRITICAL: Log name must include seed to prevent overwriting
        LOG_FILE="${LOG_DIR}/${arch}_seed${seed}.log"
        
        echo "   [Batch ${BATCH_NUM}] Launching: $arch (Seed $seed)"
        
        python "$TRAIN_SCRIPT_PATH" \
            --arch "$arch" \
            --load_params "${PROJECT_ROOT}/best_params_$arch.yaml" \
            --data_fraction 1 \
            --seed "$seed" \
            > "$LOG_FILE" 2>&1 &
            
        # Optional: Save PID if you need to kill specific jobs later
        # PIDS+=($!) 
    done

    # --- BARRIER ---
    echo "   >>> Waiting for all 3 architectures to finish for Seed ${seed}..."
    wait
    
    echo "   >>> Batch ${BATCH_NUM} Complete."
    ((BATCH_NUM++))

done

echo "======================================================================"
echo "Full Ablation Study (15 Experiments) Complete."
echo "======================================================================"