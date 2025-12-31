#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=24:00:00
#SBATCH --job-name=alpha_0-0_mach_0-1
#SBATCH --output=output_%j.txt
#SBATCH --mail-type=FAIL

module load gcc/13.3
#module load gcc/12.3
module load openmpi/5.0.3
module load imkl/2024.2.0
#module load cmake/3.21.4
#module load mkl/2019u4
module load python/3.11

export PETSC_DIR=/home/hitra/petsc-3.12.5
export PETSC_ARCH=arch-gcc12-mpich-opt
export PETSCROOT=$PETSC_DIR/$PETSC_ARCH
export GASNET_PHYSMEM_MAX='500 GB'

source /project/rrg-laurende-ab/env/CHAMPSenv_2.5.0.sh

/home/hitra/fullPotentialSolver/bin/main -f /home/hitra/fullPotentialSolver/input_file/input_file_fps_2049_alpha_0-0_mach_0-1.txt -nl 1 > /scratch/hitra/output_fps_2049_alpha_0-0_mach_0-1.txt