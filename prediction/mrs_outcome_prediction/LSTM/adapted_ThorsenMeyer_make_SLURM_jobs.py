#!/usr/bin/python
import json
import os, itertools, time, shutil, sys, stat, subprocess

# check if Python-version > 3.0
from prediction.mrs_outcome_prediction.LSTM.utils import initiate_log_files

assert (sys.version_info > (3, 0)), "This script only works with Python3!"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = '/home/users/k/klug/output/opsum/LSTM_72h'
nnet_file = 'adapted_ThorsenMeyer_LSTM.py'
batch_file = 'run_models.sh'

# make parameter dictionary
param_dict = json.load(open('./parameter_space.json'))

if __name__ == '__main__':
    # check if output directory exists
    if not os.path.exists(output_dir):
        raise ValueError('Output directory does not exist!')

    # make list of all argument combinations
    all_args = (dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values()))

    # change directory and open files for writing
    date_string = time.strftime("%Y_%m_%d_%H%M")
    working_dir = os.path.join(output_dir, date_string)
    os.makedirs(working_dir)

    initiate_log_files(working_dir)

    os.mkdir(os.path.join(working_dir, 'best_weights'))
    os.mkdir(os.path.join(working_dir, 'slurm_jobs'))
    os.mkdir(os.path.join(working_dir, 'logs'))

    # save this script to working_dir
    shutil.copy2(os.path.abspath(__file__), working_dir)
    shutil.copy2(os.path.join(script_dir, nnet_file), working_dir)

    # make batch-file
    batch_file = open(os.path.join(working_dir, 'run_models.sh'), 'w')
    batch_file.write('#!/bin/bash\n')
    batch_file.write('cd ' + output_dir + date_string + '/\n')
    batch_file.write('for f in slurm_jobs/*.sh\n')
    batch_file.write('do\n')
    batch_file.write('\tsh $f\n')
    batch_file.write('done\n')
    batch_file.close()
    # change permissions
    st = os.stat(os.path.join(working_dir, 'run_models.sh'))
    os.chmod(os.path.join(working_dir, 'run_models.sh'), st.st_mode | stat.S_IEXEC)

    # Load SLURM setup
    slurm_setup_file = open(os.path.join(script_dir, "ygdrassil_slurm_setup.txt"), "r")
    slurm_setup = slurm_setup_file.read()
    slurm_setup_file.close()


    for arg in all_args:
        shell_arg = 'srun python ' + os.path.join(script_dir, nnet_file) + ' --date_string=' + date_string \
                    + ' --output_dir=' + output_dir \
                    + ' --features_path=' + '$OPSUM_FEATURES_PATH' \
                    + ' --labels_path=' + '$OPSUM_LABELS_PATH'
        file_name = date_string
        for key in sorted(arg):
            shell_arg += ' --' + key + '=' + str(arg[key])
            file_name += '_' + str(arg[key])
            shellfile = open(os.path.join(working_dir, 'slurm_jobs', '%s.sh' % file_name), 'w')
            shellfile.write(slurm_setup)
            shellfile.write('\n cd ' + working_dir + '\n')
            shellfile.write('conda activate opsum\n')
            shellfile.write(shell_arg + '\n')
            shellfile.close()

    # # run batch-scripts
    # run_models_path = '/'.join([working_dir, r'run_models.sh'])
    # subprocess.call([run_models_path])


