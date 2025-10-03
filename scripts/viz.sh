#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name viz
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --ntasks 1
#SBATCH --mem 100G
#SBATCH --time 01:00:00

export SINGULARITY_BINDPATH="/work,/scratch,/users"

source /users/fquareng/.bashrc
micromamba activate dl
micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/viz.py



# container_path="/users/fquareng/singularity/dl_gh200.sif"
# singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/eval.py
