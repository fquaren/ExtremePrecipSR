#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name basic_emulator
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 250G
#SBATCH --time 5:00:00


export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0" 
container_path="/users/fquareng/singularity/dl_gh200.sif"
singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/train_gamma.py

# source /users/fquareng/.bashrc
# micromamba activate dl-torch
# micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/train_gamma.py

EMULATORS_TYPE=("hard" "hybrid" "soft" "none")

N_EXPERIMENTS=${#EMULATORS_TYPE[@]}
echo "Found ${N_EXPERIMENTS} experiments to evaluate."

for (( i=0; i<${N_EXPERIMENTS}; i++ )); do
    m=${EMULATORS_TYPE[i]}
    echo "--- Starting training for constraint Mode: $m ---"

    source /users/fquareng/.bashrc
    micromamba activate dl-torch
    micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/train_gamma.py \
        --constraint_mode "$m"

    echo "--- Finished evaluation for: $e ---"
    echo ""
done

echo "All evaluations complete."