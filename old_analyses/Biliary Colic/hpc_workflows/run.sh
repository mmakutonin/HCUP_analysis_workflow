#!/bin/sh
#SBATCH --time 30:00
#SBATCH -o run.out
#SBATCH -e run.err
#SBATCH -p nano -N 1
#SBATCH --mail-user=mmakutonin@gwmail.gwu.edu
#SBATCH --mail-type=ALL
#SBATCH -D /lustre/groups/meltzergrp/HCUP/biliary_colic/hpc_workflows
module load python3
. ~/hcup_env/bin/activate

#cd /lustre/groups/meltzergrp/HCUP/biliary_colic/hpc_workflows

rm -r ../tables
mkdir ../tables
rm -r ../figures/comparison plots
mkdir ../figures/comparison plots

python3 ./01-0.py && python3 ./01-1.py && \
python3 ./02-0.py && python3 ./02-1.py && python3 ./02-2.py && python3 ./02-3.py && python3 ./02-4.py && \
python3 ./03-1.py && python3 ./03-2.py && python3 ./03-3.py