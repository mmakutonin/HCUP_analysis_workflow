#!/bin/sh
#SBATCH --time 30:00
#SBATCH -o table_regen.out
#SBATCH -e table_regen.err
#SBATCH -p nano -N 1
#SBATCH --mail-user={email_address}
#SBATCH --mail-type=ALL
#SBATCH -D /lustre/groups/meltzergrp/HCUP/{analysis_name}/hpc_workflows

# Activate Conda environment to run code
. /SMHS/home/mmakutonin/miniconda3/etc/profile.d/conda.sh
# conda create -n hcup-test pandas numpy scipy statsmodels scikit-learn matplotlib
conda activate hcup-test

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

# # old functionality
# module load python3
# python3 -m pip install pandas
# python3 -m pip install numpy
# python3 -m pip install scipy
# python3 -m pip install statsmodels
# python3 -m pip install scikit-learn
# python3 -m pip install matplotlib

python3 main_regenerate_outputs_only.py