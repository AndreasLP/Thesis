#!/bin/bash
#BSUB -J CNN4[1-16]
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o batch_jobs/cnn4_%J_%I.out

module purge
module load python3/3.9.10
base_values="--lr 0.000316 --parameter_settings 18 --use_cv True --run_id $LSB_JOBID --run_index $LSB_JOBINDEX"
python3 train.py $base_values --cv_index 0 --use_dropout True
python3 train.py $base_values --cv_index 6 --use_dropout True
python3 train.py $base_values --cv_index 7 --use_dropout True
python3 train.py $base_values --cv_index 13 --use_dropout True
python3 train.py $base_values --cv_index 14 --use_dropout True
python3 train.py $base_values --cv_index 20 --use_dropout True