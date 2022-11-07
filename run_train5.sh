#!/bin/bash
#BSUB -J CNN5[1-4]
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o batch_jobs/cnn5_%J_%I.out

module purge
module load python3/3.9.10
base_values="--lr 0.000316 --parameter_settings 18 --use_cv True --run_id $LSB_JOBID --run_index $LSB_JOBINDEX --use_dropout True"
python3 train.py $base_values --cv_index 0 --use_batch_normalization True
python3 train.py $base_values --cv_index 6 --use_batch_normalization True
python3 train.py $base_values --cv_index 7 --use_batch_normalization True
python3 train.py $base_values --cv_index 13 --use_batch_normalization True
python3 train.py $base_values --cv_index 14 --use_batch_normalization True
python3 train.py $base_values --cv_index 20 --use_batch_normalization True