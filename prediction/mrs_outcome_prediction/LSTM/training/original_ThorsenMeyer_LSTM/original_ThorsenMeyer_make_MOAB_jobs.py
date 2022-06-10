#!/usr/bin/python
import os, itertools, time, shutil, sys, stat, subprocess

# check if Python-version > 3.0
assert (sys.version_info > (3, 0)), "This script only works with Python3!"

script_dir = '<PATH_FOR_THIS_SCRIPT>'
output_dir = '<PATH_FOR_OUTPUT>'
nnet_file = 'train_nnet.py'
batch_file = 'run_models.sh'

# make parameter dictionary
param_dict={}
param_dict['activation'] = ['sigmoid']
param_dict['batch'] = ['all']
param_dict['data'] = ['balanced', 'unchanged']
param_dict['dropout'] = [0.0, 0.2, 0.4, 0.6]
param_dict['layers'] = [1, 2]
param_dict['masking'] = [True]
param_dict['optimizer'] = ['RMSprop', 'Adagrad']
param_dict['outcome'] = ['dead90']
param_dict['units'] = [1, 4, 8, 16, 32, 64, 128]


# do not run this, if another script is just trying to import 'param_dict'
if __name__ == '__main__':
    # make list of all argument combinations
    all_args = (dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values()))

    # change directory and open files for writing
    date_string = time.strftime("%Y-%m-%d-%H%M")
    os.chdir(output_dir)

    os.mkdir(date_string)
    path = "".join([output_dir, date_string])
    os.chdir(path)
    os.mkdir('best_weights')
    os.mkdir('moab_jobs')
    os.mkdir('logs')

    # save this script to path
    filename = "".join([script_dir, sys.argv[0]])
    shutil.copy2(filename, path)
    shutil.copy2(script_dir + nnet_file, path)

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

    # create output files
    AUCfile = open('AUC_history_gridsearch.tsv', 'w')
    CVfile = open('CV_history_gridsearch.tsv', 'w')

    AUCheader = ['auc_train', 'auc_val', 'matthews_train', 'matthews_val', 'cv_num'] + list(param_dict.keys())

    CVheader = ['acc', 'loss', 'matthews', 'precision', 'recall', 'val_acc', 'val_loss', 'val_matthews',
    'val_precision', 'val_recall', 'cv_num'] + list(param_dict.keys())

    CVfile.write('epoch\t' + '\t'.join(CVheader) + '\n')
    AUCfile.write('\t'.join(AUCheader) + '\n')

    # close files
    AUCfile.close()
    CVfile.close()

    # open log files
    error_log = open('error.log', 'w')
    errorHeader = ['error', 'args']
    error_log.write('\t'.join(errorHeader) + '\n')
    error_log.close()

    progress_log = open('progress.log', 'w')
    progressHeader = ['completed']
    progress_log.write('\t'.join(progressHeader) + '\n')
    progress_log.close()

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
            shellfile.write('cd ' + path + '\n')
            shellfile.write(shell_arg + '\n')
            shellfile.close()

    # run batch-scripts
    run_models_path = '/'.join([path, r'run_models.sh'])
    subprocess.call([run_models_path])


