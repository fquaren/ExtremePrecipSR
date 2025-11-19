#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name convert
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --exclusive
#SBATCH --mem 0
#SBATCH --time 12:00:00

source /users/fquareng/.bashrc
micromamba activate dl-torch

DATA_ROOT="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/extremes/OPERA/patches/precip"

echo "Converting train..."
micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/convert_npz_to_npy.py ${DATA_ROOT}/train

echo "Converting validation..."
micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/convert_npz_to_npy.py ${DATA_ROOT}/validation

echo "Converting test..."
micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/convert_npz_to_npy.py ${DATA_ROOT}/test

echo "Conversion complete."