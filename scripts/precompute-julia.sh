#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name precompute-julia
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16   
#SBATCH --mem 150G            
#SBATCH --time 72:00:00      

# Load micromamba from your bashrc
source /users/fquareng/.bashrc

# Activate the Julia environment
micromamba activate julia
echo "Micromamba environment 'julia' activated."

# --- Julia Execution ---
# Run the Julia script using all allocated CPUs
# The '-t auto' flag instructs Julia to automatically use all CPUs
# allocated by SLURM (i.e., the 32 requested above).
echo "Starting Julia script..."
micromamba run -n julia julia -t auto /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/ExtremePrecipSR/src/precompute_gamma.jl