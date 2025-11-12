#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name geomeval
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 250G
#SBATCH --time 5:00:00


EMULATORS_TO_EVAL=("GammaEmulatorv2_2025-11-03_09-22-21" "GammaEmulatorv2_2025-11-03_16-53-36" "GammaEmulatorv2_2025-11-02_01-55-24" "GammaEmulatorv2_2025-11-05_00-43-26")
EMULATORS_TYPE=("hard" "hybrid" "soft" "none")

N_EXPERIMENTS=${#EMULATORS_TO_EVAL[@]}
echo "Found ${N_EXPERIMENTS} experiments to evaluate."

for (( i=0; i<${N_EXPERIMENTS}; i++ )); do
    e=${EMULATORS_TO_EVAL[i]}
    m=${EMULATORS_TYPE[i]}
    echo "--- Starting evaluation for: $e ---"
    echo "Constraint Mode: $m"

    source /users/fquareng/.bashrc
    micromamba activate dl-torch
    micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/eval_gamma.py \
        --run_dir "/scratch/fquareng/experiment_runs/$e" \
        --constraint_mode "$m"

    echo "--- Finished evaluation for: $e ---"
    echo ""
done

echo "All evaluations complete."

# export SINGULARITY_BINDPATH="/work,/scratch,/users"
# export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0" 
# container_path="/users/fquareng/singularity/dl_gh200.sif"
# singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/eval_gamma.py --run_dir /scratch/fquareng/experiment_runs/GammaEmulatorv2_2025-11-03_16-53-36
