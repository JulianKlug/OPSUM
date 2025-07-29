import os
import torch
import pandas as pd
import argparse

def convert_data_split_file_to_safe_format(input_path, output_path):
    """
    Converts a torch-saved data split file containing pandas DataFrames
    into a safe format that can be loaded regardless of pandas version.
    """
    raw_splits = torch.load(input_path, map_location='cpu')
    safe_splits = []

    for fold_X_train, fold_X_val, fold_y_train_df, fold_y_val_df in raw_splits:
        # Convert DataFrames to dictionaries (or to numpy if needed)
        fold_y_train = fold_y_train_df.to_dict(orient='list')
        fold_y_val = fold_y_val_df.to_dict(orient='list')

        safe_splits.append((fold_X_train, fold_X_val, fold_y_train, fold_y_val))

    # Save in safe format
    torch.save(safe_splits, output_path)
    print(f"✅ Converted and saved to: {output_path}")

def convert_test_data_file_to_safe_format(input_path, output_path):
    """
    Converts the test data file into a safe format.
    """
    test_X, test_y_df = torch.load(input_path, map_location='cpu')

    test_y = test_y_df.to_dict(orient='list')
    torch.save((test_X, test_y), output_path)

    print(f"✅ Converted and saved test data to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert torch-saved .pth files into version-safe formats.")
    parser.add_argument('--train', type=str, help='Path to the train .pth file to convert')
    parser.add_argument('--test', type=str, help='Path to the test .pth file to convert')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save converted files')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.train:
        train_output_path = os.path.join(args.output_dir, f'safe_{os.path.basename(args.train)}')
        convert_data_split_file_to_safe_format(args.train, train_output_path)

    if args.test:
        test_output_path = os.path.join(args.output_dir, f'safe_{os.path.basename(args.test)}')
        convert_test_data_file_to_safe_format(args.test, test_output_path)

    if not args.train and not args.test:
        print("❌ Please provide at least one of --train or --test paths.")
        parser.print_help()

if __name__ == "__main__":
    main()