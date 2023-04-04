import os
import shutil
import pandas as pd

from prediction.utils.utils import ensure_dir


def retain_only_selected_models(data_dir, model_selection_path, target_dir):

    model_selection_timestamps = pd.read_csv(model_selection_path)

    for eval_version in os.listdir(data_dir):
        eval_version_path = os.path.join(data_dir, eval_version)
        for gs_run in os.listdir(eval_version_path):
            if not gs_run.startswith('transformer_gs'):
                continue

            print(f'Processing {gs_run} of {eval_version}...')
            gs_path = os.path.join(eval_version_path, gs_run)
            for model in os.listdir(gs_path):
                if model == 'gridsearch.jsonl':
                    # move to target dir and change name to incorporate eval version and gs run
                    shutil.copyfile(os.path.join(gs_path, model),
                              os.path.join(target_dir, f'{eval_version}_{gs_run}_gridsearch.jsonl'))


                model_path = os.path.join(gs_path, model)
                model_timestamp = '_'.join(model.split('_')[3:5])
                if model_timestamp in model_selection_timestamps['timestamp'].values:
                    print(f'Keeping {model_path}')
                    ensure_dir(os.path.join(target_dir, model))
                    for file in os.listdir(model_path):
                        # skip if file already exists in target_dir
                        if os.path.exists(os.path.join(target_dir, model, file)):
                            continue
                        shutil.copyfile(os.path.join(model_path, file), os.path.join(target_dir, model, file))


def verify_selected_model_presence(target_dir, model_selection_path):
    model_selection_timestamps = pd.read_csv(model_selection_path)
    model_selection_timestamps['timestamp_found'] = False
    for timestamp in model_selection_timestamps['timestamp'].values:
        n_folds_found = 0
        for model in os.listdir(target_dir):
            if timestamp in model:
                n_folds_found += 1
        if n_folds_found == 5:
            model_selection_timestamps.loc[model_selection_timestamps['timestamp'] == timestamp, 'timestamp_found'] = True

    print(model_selection_timestamps)


if __name__ == '__main__':
    data_dir = '/mnt/data1/klug/output/transformer_evaluation'
    model_selection_path = '~/temp/model_timestamps_to_retain.csv'
    target_dir = '/mnt/hdd1/klug/output/opsum/transformer_gridsearch'

    # retain_only_selected_models(data_dir, model_selection_path, target_dir)
    verify_selected_model_presence(target_dir, model_selection_path)