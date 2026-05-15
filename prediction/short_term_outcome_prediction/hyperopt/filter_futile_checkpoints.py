import os
import shutil
import pandas as pd


def filter_futile_checkpoints(output_dir, models_to_retain_df_path, folders_to_check_prefix):
    """
    Filter out checkpoints that are not needed based on the models_to_retain_df_path
    :param output_dir: Directory containing the checkpoints
    :param models_to_retain_df_path: Path to the models_to_retain_df
    :return: None
    """
    models_to_retain_df = pd.read_csv(models_to_retain_df_path)

    models_to_retain_df['folder_to_retain'] = models_to_retain_df['file_name'].apply(lambda x: x.split('_gridsearch')[0])
    timestamps_to_retain = models_to_retain_df['timestamp'].values

    # check through all the folders matching the prefix
    for folder in os.listdir(output_dir):
        if folder.startswith(folders_to_check_prefix):
            if folder not in models_to_retain_df['folder_to_retain'].values:
                # delete all subfolders starting with "checkpoint"
                for subfolder in os.listdir(os.path.join(output_dir, folder)):
                    if subfolder.startswith('checkpoint'):
                        shutil.rmtree(os.path.join(output_dir, folder, subfolder))
            else:
                # delete all subfolders that do not contain a timestamp to retain
                for subfolder in os.listdir(os.path.join(output_dir, folder)):
                    # skip subfolder if it is not a checkpoint
                    if not subfolder.startswith('checkpoint'):
                        continue
                    matched_timestamp = False
                    for timestamp in timestamps_to_retain:
                        if timestamp in subfolder:
                            matched_timestamp = True
                            break
                    if not matched_timestamp:
                        shutil.rmtree(os.path.join(output_dir, folder, subfolder)) 
                
        else:
            print(f"Skipping folder {folder}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory containing the hyperopt run folders')
    parser.add_argument('-m', '--models_to_retain_df_path', type=str, required=True, help='Path to the models_to_retain_df')
    parser.add_argument('-p', '--folders_to_check_prefix', type=str, required=True, help='Prefix of the folders to check')
    args = parser.parse_args()

    filter_futile_checkpoints(args.output_dir, args.models_to_retain_df_path, args.folders_to_check_prefix)

