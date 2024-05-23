#!/bin/sh
#SBATCH --time 30:00
#SBATCH -o table_regen.out
#SBATCH -e table_regen.err
#SBATCH -p nano -N 1
#SBATCH --mail-user=mmakutonin@gwmail.gwu.edu
#SBATCH --mail-type=ALL
#SBATCH -D /lustre/groups/meltzergrp/HCUP/amputee_overdoses/hpc_workflows
module load python3
python3 -m pip install pandas
python3 -m pip install numpy
python3 -m pip install scipy
python3 -m pip install statsmodels
python3 -m pip install scikit-learn
python3 -m pip install matplotlib

cd /lustre/groups/meltzergrp/HCUP/amputee_overdoses/hpc_workflows

rm -r ../tables
mkdir ../tables
rm -r ../figures/comparison plots
mkdir ../figures/comparison plots

python3 ./03-1.py && python3 ./03-2.py && python3 ./03-3.py