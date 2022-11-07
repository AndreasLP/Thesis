#!/bin/bash
#BSUB -J CNN3[1-4]
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o batch_jobs/cnn3_%J_%I.out

module purge
module load python3/3.9.10

python3 train.py --lr 0.000316 --parameter_settings 18 --use_cv True --cv_index 0 --run_id $LSB_JOBID --run_index $LSB_JOBINDEX
python3 train.py --lr 0.000316 --parameter_settings 18 --use_cv True --cv_index 6 --run_id $LSB_JOBID --run_index $LSB_JOBINDEX
python3 train.py --lr 0.000316 --parameter_settings 18 --use_cv True --cv_index 7 --run_id $LSB_JOBID --run_index $LSB_JOBINDEX
python3 train.py --lr 0.000316 --parameter_settings 18 --use_cv True --cv_index 13 --run_id $LSB_JOBID --run_index $LSB_JOBINDEX
python3 train.py --lr 0.000316 --parameter_settings 18 --use_cv True --cv_index 14 --run_id $LSB_JOBID --run_index $LSB_JOBINDEX
python3 train.py --lr 0.000316 --parameter_settings 18 --use_cv True --cv_index 20 --run_id $LSB_JOBID --run_index $LSB_JOBINDEX