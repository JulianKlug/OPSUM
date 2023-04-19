#!/bin/bash
#SBATCH --job-name=opsum_compute_shap    # Job name
#SBATCH --partition=public-gpu
#SBATCH --gres=gpu:turing:1                    # Number of GPUs
#SBATCH --ntasks=1
#SBATCH --mem=100G                     # Job memory request
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=/home/users/k/klug/logs/opsum/opsum_shap_%j.log   # Standard output and error log

ulimit -S -n 131072 # setting "soft" limit number of file descriptor per processes
ulimit -S -u 1546461 # setting soft limit number of processes per user

module load Anaconda3

source /home/users/k/klug/.bashrc

export OPSUM_FEATURES_PATH="/home/users/k/klug/data/opsum/gsu_prepro_01012023_233050/preprocessed_features_01012023_233050.csv"
export OPSUM_LABELS_PATH="/home/users/k/klug/data/opsum/gsu_prepro_01012023_233050/preprocessed_outcomes_01012023_233050.csv"
export OPSUM_LOGS_PATH="/home/users/k/klug/logs/opsum/opsum_shap_${SLURM_JOB_ID}.log"

conda activate shap
export PYTHONPATH="${PYTHONPATH}:/home/users/k/klug/opsum"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

cd /home/users/k/klug/output/opsum/LSTM_72h/3M_mRS_0-2/2023_01_02_1057
srun python /home/users/k/klug/opsum/prediction/outcome_prediction/Transformer/inference/compute_shap_explanations_over_time.py --output_dir=/home/users/k/klug/output/opsum/transformer_72h/shap_values --features_path=$OPSUM_FEATURES_PATH --labels_path=$OPSUM_LABELS_PATH --outcome="3M mRS 0-2" --model_weights_path /home/users/k/klug/output/opsum/transformer_72h/shap_values/opsum_transformer_epoch=14_val_auroc=0.9222.ckpt --model_config_path /home/users/k/klug/output/opsum/transformer_72h/shap_values/hyperopt_selected_transformer_20230402_184459.json
cp $OPSUM_LOGS_PATH /home/users/k/klug/output/opsum/transformer_72h/shap_values
