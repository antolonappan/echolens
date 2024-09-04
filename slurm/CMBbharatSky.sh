#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=8
#SBATCH --ntasks=250
#SBATCH --cpus-per-task=1
#SBATCH -J LBIRD-FG0
#SBATCH -o out/fg0.out
#SBATCH -e out/fg0.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


module load python
conda activate cosmos
cd /home/alonappan/workspace/echolens/slurm

mpirun -np $SLURM_NTASKS python jobs.py config_template.ini -n $SLURM_NTASKS