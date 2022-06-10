import json
import os


def initiate_log_files(output_dir):
    #  load parameters from json file
    param_dict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'training/parameter_space.json')))

    # create output files
    AUCfile_path = os.path.join(output_dir, 'AUC_history_gridsearch.tsv')
    if not os.path.exists(AUCfile_path):
        AUCfile = open(AUCfile_path, 'w')

        AUCheader = ['auc_train', 'auc_val', 'matthews_train', 'matthews_val', 'cv_num'] + list(param_dict.keys())
        AUCfile.write('\t'.join(AUCheader) + '\n')

        # close files
        AUCfile.close()

    CVfile_path = os.path.join(output_dir, 'CV_history_gridsearch.tsv')
    if not os.path.exists(CVfile_path):
        CVfile = open(CVfile_path, 'w')

        CVheader = ['accuracy', 'loss', 'matthews', 'precision', 'recall', 'val_accuracy', 'val_loss', 'val_matthews',
        'val_precision', 'val_recall', 'cv_num'] + list(param_dict.keys())

        CVfile.write('epoch\t' + '\t'.join(CVheader) + '\n')

        CVfile.close()

    # open log files
    error_log_path = os.path.join(output_dir,'error.log')
    if not os.path.exists(error_log_path):
        error_log = open(error_log_path, 'w')
        errorHeader = ['error', 'args']
        error_log.write('\t'.join(errorHeader) + '\n')
        error_log.close()

    progress_log_path = os.path.join(output_dir,'progress.log')
    if not os.path.exists(progress_log_path):
        progress_log = open(progress_log_path, 'w')
        progressHeader = ['completed']
        progress_log.write('\t'.join(progressHeader) + '\n')
        progress_log.close()

    return 0
