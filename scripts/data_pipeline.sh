#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name datapipeline
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --ntasks 1
#SBATCH --mem 500G
#SBATCH --time 72:00:00

# Force single-threading for numpy, etc.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export SINGULARITY_BINDPATH="/work,/scratch,/users"

source /users/fquareng/.bashrc

micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/data/generate_patch_metadata.py
micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/data/preprocess_data_parallel.py
