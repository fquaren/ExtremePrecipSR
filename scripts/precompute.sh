#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name precompute
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 350G
#SBATCH --time 10:00:00

# export SINGULARITY_BINDPATH="/work,/scratch,/users"
# container_path="/users/fquareng/singularity/dl_gh200.sif"
# singularity exec "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/precompute_gamma_with_CC.py

source /users/fquareng/.bashrc
micromamba activate dl
micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/precompute_gamma_with_CC.py
