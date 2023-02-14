#!/usr/bin/python
import os
from prediction.utils.SLURM_job_generator import generate_SLURM_jobs


script_dir = os.path.dirname(os.path.abspath(__file__))
# output_dir = '/home/users/k/klug/output/opsum/transformer_72h'
output_dir = '/Users/jk1/temp/opsum_prediction_output/transformer_test'

nnet_file = 'transformer_trainer.py'
batch_file = 'run_models.sh'

param_path = os.path.join(script_dir, 'parameter_space.json')

if __name__ == '__main__':
    generate_SLURM_jobs(output_dir, param_path, script_dir, nnet_file_name=nnet_file, batch_file_name=batch_file)

