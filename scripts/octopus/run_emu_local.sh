#!/bin/bash

# --- USER CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ExtremePrecipSR"
SCRIPT_PATH="${PROJECT_ROOT}/src/train_gamma.py"
LOG_DIR="${PROJECT_ROOT}/logs/GammaEmulators_$(date +%Y%m%d_%H%M%S)"

# Create log directory to ensure reproducibility
mkdir -p "$LOG_DIR"

# --- EXPERIMENT SETUP ---
EMULATORS_TYPE=("hard" "hybrid" "soft" "none")
ARCHITECTURE_TYPE=("Vanilla" "Attention")

N_EXPERIMENTS=${#EMULATORS_TYPE[@]}
N_ARCHS=${#ARCHITECTURE_TYPE[@]}

echo "----------------------------------------------------"
echo "Machine: $(hostname)"
echo "Found ${N_EXPERIMENTS} Emulator types X ${N_ARCHS} Architectures."
echo "Total runs: $((N_EXPERIMENTS * N_ARCHS))"
echo "Logs will be saved to: $LOG_DIR"
echo "----------------------------------------------------"

# --- EXECUTION LOOP ---
# I simplified the C-style loop to a standard bash array iteration for readability
for m in "${EMULATORS_TYPE[@]}"; do
    for a in "${ARCHITECTURE_TYPE[@]}"; do
        
        LOG_FILE="${LOG_DIR}/train_${m}_${a}.log"
        
        echo "Starting: Mode=$m | Arch=$a"
        echo "Logging to: $LOG_FILE"

        # We use 'micromamba run' to execute safely within the environment.
        # stdout and stderr are redirected to the log file.
        source /home/fquareng/.bashrc
        micromamba run -n dl python "$SCRIPT_PATH" \
            --constraint_mode "$m" \
            --arch "$a" \
            > "$LOG_FILE" 2>&1

        # Check exit status of the last command
        if [ $? -eq 0 ]; then
            echo "SUCCESS: $m, $a"
        else
            echo "FAILURE: $m, $a - Check log for details."
        fi
        echo "----------------------------------------------------"
        
    done
done

echo "All training experiments complete."