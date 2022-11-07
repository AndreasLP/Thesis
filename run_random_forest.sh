#!/bin/bash
#BSUB -J RF[1-12]%6
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -q hpc
#BSUB -W 30
#BSUB -R "rusage[mem=3GB]"
#BSUB -o batch_jobs/random_forest_%J_%I.out

module purge
module load python3/3.9.10

python3 random_forest.py $LSB_JOBINDEX