# Dataset structure 
Main files
- {dataset}_patient_characteristics.csv : table 1
- preprocessed_features_{version}.csv : feature dataset of selected variables over the 72 first hours after admission
- preprocessed_outcomes_{version}.csv : outcomes for all patients

Subfolders: 
- data_visualisation: visual representation of features over time for each patient
- logs_{version}: logs of the preprocessing

### Feature dataset structure

| relative_sample_date_hourly_cat | case_admission_id | sample_label | source | value |
|----------------------------------|-------------------|--------------|--------|-------|
| 0 | 100023_4784 | ALAT | EHR_pop_imputed | -0.15919332 |

- relative_sample_date_hourly_cat: Time point relative to admission (0 = admission hour)
    - observations are binned in timebins of an arbitrary size (e.g.: all observations realised at between the 4th and 5th hour after admission are binned together)
- case_admission_id: Unique patient identifier
- sample_label: Feature name (e.g., ALAT)
- source: Data source/imputation method
    - EHR: from EHR
    - EHR_locf_imputed: imputation by carrying forward last EHR observation
    - EHR_pop_imputed: imputation from overall population EHR data
    - EHR_pop_imputed_locf_imputed: imputation by carrying forward last population imputation
    - stroke_registry: from stroke registry
    - stroke_registry_locf_imputed: imputation by carrying forward last registry observation
    - stroke_registry_pop_imputed: imputation from overall population registry data
    - stroke_registry_pop_imputed_locf_imputed: imputation by carrying forward last population imputation
    - notes: from clinical notes (manually extracted data)
    - notes_locf_imputed: imputation by carrying forward last observation
    - notes_pop_imputed: imputation by overall population notes data
    - notes_pop_imputed_locf_imputed: imputation by carrying forward last population imputation
- value: Normalized feature value

## Reproducibility 

A dataset folder can be generated using following preprocessing pipeline: https://github.com/JulianKlug/OPSUM/blob/main/preprocessing

Preprocessing steps are explained here: https://github.com/JulianKlug/OPSUM/blob/main/preprocessing/geneva_stroke_unit_preprocessing/readme.md

Visualisations of the feature dataset can be created using: https://github.com/JulianKlug/OPSUM/blob/main/data_visualisation/preprocessed_patient_visualisation.ipynb

> Klug, J., Leclerc, G., Dirren, E. et al. Machine learning for early dynamic prediction of functional outcome after stroke. Commun Med 4, 232 (2024). https://doi.org/10.1038/s43856-024-00666-w
