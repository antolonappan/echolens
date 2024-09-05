#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=4
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH -J ECHO
#SBATCH -o sky.out
#SBATCH -e sky.err
#SBATCH --time=00:30:00

module load python
conda activate cosmos
cd /home/alonappan/workspace/echolens/slurm

mpirun -np $SLURM_NTASKS python jobs.py config_template.ini -n $SLURM_NTASKS