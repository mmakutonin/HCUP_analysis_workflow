#!/bin/sh
#SBATCH --time 30:00
#SBATCH -o run.out
#SBATCH -e run.err
#SBATCH -p nano -N 1
#SBATCH --mail-user=mmakutonin@gwmail.gwu.edu
#SBATCH --mail-type=ALL
#SBATCH -D /lustre/groups/meltzergrp/HCUP/{analysis_name}/hpc_workflows

# # Activate Conda environment to run code
# . /SMHS/home/mmakutonin/miniconda3/etc/profile.d/conda.sh
# # conda create -n hcup pandas numpy scipy statsmodels scikit-learn matplotlib
# conda activate hcup

# # to prepare the directory for files
# cd /lustre/groups/meltzergrp/HCUP/{analysis_name}
# rm -r hpc_workflows
# in a separate CMD (not on pegasus)
# scp -r HCUP/{analysis_name}/hpc_workflows mmakutonin@pegasus.arc.gwu.edu:/lustre/groups/meltzergrp/HCUP/{analysis_name}

# to run
# sbatch /lustre/groups/meltzergrp/HCUP/{analysis_name}/hpc_workflows/run.sh

# to check available nodes and status
# sinfo
# squeue

# old functionality
module load python3
python3 -m pip install pandas
python3 -m pip install numpy
python3 -m pip install scipy
python3 -m pip install statsmodels
python3 -m pip install scikit-learn
python3 -m pip install matplotlib

rm -r ../tables
mkdir ../tables
rm -r ../figures/comparison plots
mkdir ../figures/comparison plots

python3 ./01-0.py && python3 ./01-1.py && \
python3 ./02-0.py && python3 ./02-1.py && python3 ./02-2.py && python3 ./02-3.py && python3 ./02-4.py && \
python3 ./03-1.py && python3 ./03-2.py && python3 ./03-3.py