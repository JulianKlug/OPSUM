

def detect_overwritten_patient_ids(df, eds_df, verbose=False):
    """
    Detects if patient_id is overwritten by eds_final_patient_id
    :param df:
    :param eds_df:
    :param verbose:
    :return:
    """
    eds_df.drop_duplicates(subset=['patient_id', 'eds_end_4digit', 'eds_manual', 'patient_id_manual'], inplace=True)
    df.drop_duplicates(subset=['patient_id', 'eds_end_4digit', 'eds_manual', 'patient_id_manual'], inplace=True)

    if len(eds_df[eds_df.patient_id != eds_df.eds_final_patient_id]) == 0:
        if verbose:
            print('Perfect match between patient_id and eds_final_patient_id in eds, thus overwriting does not matter.')
        return 0
    else:
        # Overlap between patient_id and eds_final_patient_id should not be perfect before EDS filtering (because of extraction errors)
        n_missmatch = len(df[df.patient_id != df.eds_final_patient_id])
        if n_missmatch == 0:
            if verbose:
                print('No missmatch between patient_id and eds_final_patient_id in df.')
            print('WARNING: patient_id in df was probably overwritten by eds_final_patient_id.')
        else:
            if verbose:
                print('Missmatch between patient_id and eds_final_patient_id in df. Thus overwriting of patient_id is '
                      'unlikely')

        return n_missmatch


if __name__ == '__main__':
    import argparse, os
    import pandas as pd
    from preprocessing.geneva_stroke_unit_preprocessing.variable_assembly.variable_database_assembly import load_data_from_main_dir

    parser = argparse.ArgumentParser(description='Detects if patient_id is overwritten by eds_final_patient_id')
    parser.add_argument('-d', '--data_path', help='Path to extracted_data', required=True)
    parser.add_argument('-o', '--output_path', help='Path to output logs', required=True)

    args = parser.parse_args()

    eds_df = pd.read_csv(os.path.join(args.data_path, 'eds_j1.csv'), delimiter=';', encoding='utf-8',
                         dtype=str)

    print('-- Detection of overwritten patient_id in eds data --')
    eds_n_missmatch = detect_overwritten_patient_ids(eds_df, eds_df, verbose=True)

    print('-- Detection of overwritten patient_id in lab data --')
    lab_file_start = 'labo'
    lab_df = load_data_from_main_dir(args.data_path, lab_file_start)
    lab_n_missmatch = detect_overwritten_patient_ids(lab_df, eds_df, verbose=True)

    print('-- Detection of overwritten patient_id in scale data --')
    scales_file_start = 'scale'
    scales_df = load_data_from_main_dir(args.data_path, scales_file_start)
    scales_n_missmatch = detect_overwritten_patient_ids(scales_df, eds_df, verbose=True)

    print('-- Detection of overwritten patient_id in ventilation data --')
    ventilation_file_start = 'ventilation'
    ventilation_df = load_data_from_main_dir(args.data_path, ventilation_file_start)
    ventilation_n_missmatch = detect_overwritten_patient_ids(ventilation_df, eds_df, verbose=True)

    print('-- Detection of overwritten patient_id in patientvalue data --')
    vitals_file_start = 'patientvalue'
    vitals_df = load_data_from_main_dir(args.data_path, vitals_file_start)
    vitals_n_missmatch = detect_overwritten_patient_ids(vitals_df, eds_df, verbose=True)

    # assemble results in a dataframe
    results_df = pd.DataFrame(columns=['eds_n_missmatch', 'lab_n_missmatch', 'scales_n_missmatch', 'ventilation_n_missmatch', 'vitals_n_missmatch', 'comment'])
    results_df.loc[0] = [eds_n_missmatch, lab_n_missmatch, scales_n_missmatch, ventilation_n_missmatch, vitals_n_missmatch,
                         'Number of missmatchs between patient_id and eds_final_patient_id in EHR data']
    results_df.to_csv(os.path.join(args.output_path, 'overwritten_patient_id_detection.csv'), index=False)



