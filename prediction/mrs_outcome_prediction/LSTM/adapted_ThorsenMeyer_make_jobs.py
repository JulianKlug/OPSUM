#!/usr/bin/python
import json
import os, itertools, time, shutil, sys, stat, subprocess

# check if Python-version > 3.0
from prediction.mrs_outcome_prediction.LSTM.utils import initiate_log_files

assert (sys.version_info > (3, 0)), "This script only works with Python3!"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = '/Users/jk1/temp/opsum_prediction_output/LSTM_72h'
nnet_file = 'adapted_ThorsenMeyer_LSTM.py'
batch_file = 'run_models.sh'

# make parameter dictionary
param_dict = json.load(open('./parameter_space.json'))

# do not run this, if another script is just trying to import 'param_dict'
if __name__ == '__main__':
    # make list of all argument combinations
    all_args = (dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values()))

    # change directory and open files for writing
    date_string = time.strftime("%Y_%m_%d_%H%M")
    working_dir = os.path.join(output_dir, date_string)
    os.makedirs(working_dir)

    initiate_log_files(working_dir)

    os.chdir(working_dir)
    os.mkdir('best_weights')
    os.mkdir('moab_jobs')
    os.mkdir('logs')

    # save this script to working_dir
    shutil.copy2(os.path.abspath(__file__), working_dir)
    shutil.copy2(os.path.join(script_dir, nnet_file), working_dir)

    # make batch-file
    batch_file = open('run_models.sh', 'w')
    batch_file.write('#!/bin/bash\n')
    batch_file.write('cd ' + output_dir + date_string + '/\n')
    batch_file.write('for f in moab_jobs/*.sh\n')
    batch_file.write('do\n')
    batch_file.write('\tqsub $f\n')
    batch_file.write('done\n')
    batch_file.close()
    # change permissions
    st = os.stat('run_models.sh')
    os.chmod('run_models.sh', st.st_mode | stat.S_IEXEC)

    CMD = '''#!/bin/bash
     #PBS -l nodes=1:ppn=1
     #PBS -l mem=300gb
     #PBS -l walltime=96:00:00
     #PBS -e logs/$PBS_JOBID.err
     #PBS -o logs/$PBS_JOBID.log
         '''
    for arg in all_args:
        shell_arg = 'python ' + nnet_file + ' --date_string=' + date_string
        file_name = date_string
        for key in sorted(arg):
            shell_arg += ' --' + key + '=' + str(arg[key])
            file_name += '_' + str(arg[key])
            shellfile = open('moab_jobs/%s.sh' % file_name, 'w')
            shellfile.write(CMD)
            shellfile.write('cd ' + working_dir + '\n')
            shellfile.write(shell_arg + '\n')
            shellfile.close()

    # # run batch-scripts
    # run_models_path = '/'.join([working_dir, r'run_models.sh'])
    # subprocess.call([run_models_path])


