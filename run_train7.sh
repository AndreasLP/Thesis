#!/bin/bash
#BSUB -J CNN7[27-30]
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -o batch_jobs/cnn7_%J_%I.out

module purge
module load python3/3.9.10

base_values="--lr 0.000316 --use_cv True --run_id $LSB_JOBID --run_index $LSB_JOBINDEX --use_dropout True"
python3 train.py $base_values --cv_index 0 --parameter_settings $LSB_JOBINDEX
python3 train.py $base_values --cv_index 6 --parameter_settings $LSB_JOBINDEX
python3 train.py $base_values --cv_index 7 --parameter_settings $LSB_JOBINDEX
python3 train.py $base_values --cv_index 13 --parameter_settings $LSB_JOBINDEX
python3 train.py $base_values --cv_index 14 --parameter_settings $LSB_JOBINDEX
python3 train.py $base_values --cv_index 20 --parameter_settings $LSB_JOBINDEX