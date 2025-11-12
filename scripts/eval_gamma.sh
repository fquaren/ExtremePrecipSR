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


# source /users/fquareng/.bashrc
# micromamba activate dl-torch
# micromamba run -n dl-torch python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/eval_gamma.py --run_dir /scratch/fquareng/experiment_runs/GammaEmulatorv2_2025-11-05_00-43-26

export SINGULARITY_BINDPATH="/work,/scratch,/users"
export SINGULARITYENV_LD_PRELOAD="/opt/hpcx/ucc/lib/libucc.so.1:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucs.so.0" 
container_path="/users/fquareng/singularity/dl_gh200.sif"
singularity exec --nv "$container_path" python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/eval_gamma.py --run_dir /scratch/fquareng/experiment_runs/GammaEmulatorv2_2025-11-05_00-43-26
