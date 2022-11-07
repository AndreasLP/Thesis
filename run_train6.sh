#!/bin/bash
#BSUB -J CNN6[1-8]
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o batch_jobs/cnn6_%J_%I.out

module purge
module load python3/3.9.10
base_values="--lr 0.000316 --parameter_settings 18 --use_cv True --run_id $LSB_JOBID --run_index $LSB_JOBINDEX --use_dropout True"
python3 train.py $base_values --cv_index 0 --seed $LSB_JOBINDEX
python3 train.py $base_values --cv_index 6 --seed $LSB_JOBINDEX
python3 train.py $base_values --cv_index 7 --seed $LSB_JOBINDEX
python3 train.py $base_values --cv_index 13 --seed $LSB_JOBINDEX
python3 train.py $base_values --cv_index 14 --seed $LSB_JOBINDEX
python3 train.py $base_values --cv_index 20 --seed $LSB_JOBINDEX