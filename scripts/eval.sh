#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name geomeval
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 250G
#SBATCH --time 1:00:00

export SINGULARITY_BINDPATH="/work,/scratch,/users"

# source /users/fquareng/.bashrc
# micromamba activate dl
# micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/eval.py --run_dir /scratch/fquareng/experiment_runs/GammaEmulatorv1_2025-10-27_16-42-22

container_path="/users/fquareng/singularity/dl_gh200.sif"
singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/eval.py --run_dir /scratch/fquareng/experiment_runs/GammaEmulatorv1_2025-10-27_16-42-22
