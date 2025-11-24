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
#SBATCH --cpus-per-task 16
#SBATCH --mem 250G
#SBATCH --time 24:00:00


# export SINGULARITY_BINDPATH="/work,/scratch,/users"
# export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0" 
# container_path="/users/fquareng/singularity/dl_gh200.sif"
# singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/train_gamma.py

# source /users/fquareng/.bashrc
# micromamba activate dl-torch
# micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/train_gamma.py

EMULATORS_TYPE=("hybrid" "hard" "soft" "none")
ARCHITECTURE_TYPE=("Vanilla" "Attention")

N_EXPERIMENTS=${#EMULATORS_TYPE[@]}
N_ARCHS=${#ARCHITECTURE_TYPE[@]}
echo "Found ${N_EXPERIMENTS} X ${N_ARCHS} experiments to train."

for (( i=0; i<${N_EXPERIMENTS}; i++ )); do
    m=${EMULATORS_TYPE[i]}
    for a in "${ARCHITECTURE_TYPE[@]}"; do
        echo "--- Starting training for constraint Mode: $m, Architecture: $a ---"

        source /users/fquareng/.bashrc
        micromamba activate dl-torch
        micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/train_gamma.py \
            --constraint_mode "$m" --arch "$a"
        # singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/train_gamma.py \
        #     --constraint_mode "$m" --arch "$a"

        echo "--- Finished training for: $m, $a ---"
        echo ""
    done
done

echo "All training complete."