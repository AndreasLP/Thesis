#!/bin/bash
#BSUB -J CNN10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -o batch_jobs/cnn10_%J_%I.out

module purge
module load python3/3.9.10

base_values="--lr 0.000316 --use_cv True --run_id $LSB_JOBID --run_index $LSB_JOBINDEX --use_dropout True \
--parameter_settings 20 --activation_function elu --weight_decay 3.1623e-07"
python3 train.py $base_values --cv_index 0 --use_lr_schedule True
python3 train.py $base_values --cv_index 6 --use_lr_schedule True
python3 train.py $base_values --cv_index 7 --use_lr_schedule True
python3 train.py $base_values --cv_index 13 --use_lr_schedule True
python3 train.py $base_values --cv_index 14 --use_lr_schedule True
python3 train.py $base_values --cv_index 20 --use_lr_schedule True