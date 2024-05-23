#!/bin/sh
#SBATCH --time 30:00
#SBATCH -o run.out
#SBATCH -e run.err
#SBATCH -p large-gpu -N 1
#SBATCH --mail-user=mmakutonin@gwmail.gwu.edu
#SBATCH --mail-type=ALL
#SBATCH -D /lustre/groups/meltzergrp/HCUP/corneal_abrasion/hpc_workflows
module load python3
python3 -m pip install pandas
python3 -m pip install numpy
python3 -m pip install scipy
python3 -m pip install statsmodels
python3 -m pip install scikit-learn
python3 -m pip install matplotlib

cd /lustre/groups/meltzergrp/HCUP/corneal_abrasion/hpc_workflows

rm -r ../tables
mkdir ../tables
rm -r ../figures/comparison plots
mkdir ../figures/comparison plots

python ./01-0.py && python ./01-1.py && \
python ./02-0.py && python ./02-1.py && python ./02-2.py && python ./02-3.py && python ./02-4.py && \
python ./03-1.py && python ./03-2.py && python ./03-3.py