#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name geomeval
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --ntasks 1
#SBATCH --mem 250G
#SBATCH --time 2:00:00

export SINGULARITY_BINDPATH="/work,/scratch,/users"

source /users/fquareng/.bashrc
micromamba activate dl
micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/eval.py --run_dir /scratch/fquareng/GammaEmulatorv0

# container_path="/users/fquareng/singularity/dl_gh200.sif"
# singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/eval.py
