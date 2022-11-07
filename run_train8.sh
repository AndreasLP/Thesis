#!/bin/bash
#BSUB -J CNN8[1-5]
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -o batch_jobs/cnn8_%J_%I.out

module purge
module load python3/3.9.10

base_values="--lr 0.000316 --use_cv True --run_id $LSB_JOBID --run_index $LSB_JOBINDEX --use_dropout True --parameter_settings 20"
python3 train.py $base_values --cv_index 0
python3 train.py $base_values --cv_index 6
python3 train.py $base_values --cv_index 7
python3 train.py $base_values --cv_index 13
python3 train.py $base_values --cv_index 14
python3 train.py $base_values --cv_index 20