#!/bin/sh
#SBATCH --time 30:00
#SBATCH -o run.out
#SBATCH -e run.err
#SBATCH -p nano -N 1
#SBATCH --mail-user=mmakutonin@gwmail.gwu.edu
#SBATCH --mail-type=ALL
#SBATCH -D /lustre/groups/meltzergrp/HCUP/biliary_colic/hpc_find_uncomplicated_cases_drops
module load python3
python3 -m pip install pandas
python3 -m pip install numpy
python3 -m pip install scipy
python3 -m pip install statsmodels
python3 -m pip install scikit-learn
python3 -m pip install matplotlib

cd /lustre/groups/meltzergrp/HCUP/biliary_colic/hpc_find_uncomplicated_cases_drops

python3 ./01-0.py && python3 ./01-1.py && \
python3 ./02-0.py && python3 ./02-1.py && python3 ./02-2.py && python3 ./02-3.py && python3 ./02-4.py && \
python3 ./03-1.py