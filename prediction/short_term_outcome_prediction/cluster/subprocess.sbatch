#!/bin/bash
#SBATCH --job-name=opsum_short_subprocess    # Job name
#SBATCH --partition=public-gpu
#SBATCH --gres=gpu:turing:1                    # Number of GPUs
#SBATCH --ntasks=1
#SBATCH --mem=100G                     # Job memory request
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=/home/users/k/klug/logs/short_term_opsum/opsum_short_subprocess_%j.log   # Standard output and error log

ulimit -S -n 131072 # setting "soft" limit number of file descriptor per processes
ulimit -S -u 1546461 # setting soft limit number of processes per user

module load Anaconda3

source /home/users/k/klug/.bashrc

export OPSUM_LOGS_PATH="/home/users/k/klug/logs/short_term_opsum/opsum_short_subprocess_${SLURM_JOB_ID}.log"

conda deactivate
conda activate opsum
export PYTHONPATH="${PYTHONPATH}:/home/users/k/klug/opsum"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

srun python $subprocess_py_file_path -d $data_splits_path -o $output_folder -t $trial_name -c $gridsearch_config_path -g $use_gpu -spwd $storage_pwd -sport $storage_port -shost $storage_host
cp $OPSUM_LOGS_PATH $output_folder
