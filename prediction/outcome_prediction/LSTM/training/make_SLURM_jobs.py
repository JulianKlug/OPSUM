#!/usr/bin/python
import json
import os, itertools, time, shutil, sys, stat, subprocess
from prediction.outcome_prediction.LSTM.training.utils import initiate_log_files
from prediction.utils.utils import ensure_dir

"""
This code is adapted from an original work by Thorsen-Meyer et al.
Reference: Thorsen-Meyer H-C, Nielsen AB, Nielsen AP, et al. Dynamic and explainable machine learning prediction of mortality in patients in the intensive care unit: a retrospective study of high-frequency data in electronic patient records. Lancet Digital Health 2020; published online March 12. https://doi.org/10.1016/ S2589-7500(20)30018-2.
"""


# check if Python-version > 3.0
assert (sys.version_info > (3, 0)), "This script only works with Python3!"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = '/home/users/k/klug/output/opsum/LSTM_72h'

nnet_file = 'LSTM_trainer.py'
batch_file = 'run_models.sh'

# make parameter dictionary
param_dict = json.load(open('parameter_space.json'))

if __name__ == '__main__':
    # check if output directory exists
    if not os.path.exists(output_dir):
        raise ValueError('Output directory does not exist!')

    # make list of all argument combinations
    all_args = (dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values()))

    # change directory and open files for writing
    date_string = time.strftime("%Y_%m_%d_%H%M")
    outcome_dir = os.path.join(output_dir, '_'.join(param_dict['outcome']).replace(' ', '_'))
    ensure_dir(outcome_dir)
    working_dir = os.path.join(outcome_dir, date_string)
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
    batch_file.write('cd ' + working_dir + '\n')
    batch_file.write('for f in slurm_jobs/*.sbatch\n')
    batch_file.write('do\n')
    batch_file.write('\tsbatch "$f"\n')
    batch_file.write('done\n')
    batch_file.close()
    # change permissions
    st = os.stat(os.path.join(working_dir, 'run_models.sh'))
    os.chmod(os.path.join(working_dir, 'run_models.sh'), st.st_mode | stat.S_IEXEC)

    # Load SLURM setup
    slurm_setup_path = os.path.join(script_dir, "ygdrassil_slurm_setup.txt")
    slurm_setup_file = open(slurm_setup_path, "r")
    slurm_setup = slurm_setup_file.read()
    slurm_setup_file.close()
    shutil.copy2(slurm_setup_path, working_dir)


    for arg in all_args:
        shell_arg = 'srun python ' + os.path.join(script_dir, nnet_file) + ' --date_string=' + date_string \
                    + ' --output_dir=' + working_dir \
                    + ' --features_path=' + '$OPSUM_FEATURES_PATH' \
                    + ' --labels_path=' + '$OPSUM_LABELS_PATH'
        file_name = date_string
        for key in sorted(arg):
            option = str(arg[key])
            if ' ' in option:
                option = '"' + option + '"'
            shell_arg += ' --' + key + '=' + option
            file_name += '_' + str(arg[key])
        shellfile = open(os.path.join(working_dir, 'slurm_jobs', '%s.sbatch' % file_name), 'w')
        shellfile.write(slurm_setup)
        shellfile.write('\ncd ' + working_dir + '\n')
        shellfile.write(shell_arg + '\n')
        # copy logs to log dir
        shellfile.write('cp $OPSUM_LOGS_PATH ' + os.path.join(working_dir, 'logs') + '\n')
        shellfile.close()

    # run batch-scripts
    run_models_path = '/'.join([working_dir, r'run_models.sh'])
    subprocess.call([run_models_path])



