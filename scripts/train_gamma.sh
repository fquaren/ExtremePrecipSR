#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name emulator
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 24
#SBATCH --mem 350G
#SBATCH --time 72:00:00

# --- Environment Setup ---
source /users/fquareng/.bashrc
micromamba activate dl-torch

# --- 1. Data Generation Step ---
# Only runs once before the training loop starts
echo "--- Running Offline Data Generation (MixUp) ---"
micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/data/mixup_dataset.py

# --- 2. Define Experiment Configurations ---
# We use parallel arrays to define specific pairs of (Architecture, Constraint Mode)
# 1. Attention with hard constraints
# 2. Attention with soft constraints
# 3. Attention no constraints
# 4. Vanilla no constraints

ARCHS=("Attention" "Attention" "Attention" "Vanilla")
MODES=("hard"      "soft"      "none"      "none")

# Get number of experiments (should be 4)
N_EXPS=${#ARCHS[@]}

echo "Found ${N_EXPS} specific experiments to train."

# --- 3. Training Loop ---
for (( i=0; i<${N_EXPS}; i++ )); do
    curr_arch=${ARCHS[i]}
    curr_mode=${MODES[i]}

    echo "=========================================================="
    echo "Experiment $((i+1))/${N_EXPS}: Architecture=${curr_arch}, Mode=${curr_mode}"
    echo "=========================================================="

    micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/train_gamma.py \
        --constraint_mode "$curr_mode" --arch "$curr_arch" --data_percentage 10

    echo "--- Finished Experiment $((i+1)) ---"
    echo ""
done

echo "All training complete."