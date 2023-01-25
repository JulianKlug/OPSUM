### A. Feature preprocessing

1. Select variables
   - Path: selected_variables.xlsx

2. Preprocessing & database creation
   - Path: ./database_assembly/database_assembly.py
   - Details: 
     - Restrict to patient selection (& filter out patients with no data extracted from admission/discharge notes)
     - Preprocess EHR and stroke registry variables
       - Preprocessing involves restriction to plausible ranges: ./possible_ranges_for_variables.xlsx
     - Restrict to variable selection
     - Assemble database from lab / monitoring / data extracted from notes

3. Transform timestamps to relative timestamps from first measure
   - Path: ./database_assembly/relative_timestamps.py
   - Note: reference timepoint is the first occurrence of a monitoring variable (i.e. a variable from the EHR that is not a lab measurement)

4. Exclude patients with data sampled in a time window < 12h
   - Excluded patients are logged under: 'excluded_patients_with_sampling_range_too_short.tsv' 
   - Path: ./database_assembly/relative_timestamps.py

5. Restrict to desired time range 
   - Chosen time range: 72h
   - Path: ./database_assembly/relative_timestamps.py

6. Encoding categorical variables
   - Categorical variables are one hot encoded 
   - Path: ./encoding_categorical_variables/encoding_categorical_variables.py

7. Resampling to selected frequency: _hourly_
    - Downsampling selected features to hourly median/max/min values: ['NIHSS', 'oxygen_saturation', 'systolic_blood_pressure', 'diastolic_blood_pressure', 'mean_blood_pressure', 'heart_rate', 'respiratory_rate']
    - All other features are downsampled to hourly median
    - Path: ./resample_to_time_bins/resample_to_hourly_features.py

8. Imputation of missing values
     - Missing values are imputed by last observation carried forward (LOCF). 
     - Population medians in the datasets are used for missing values occurring before the first actual measurement.
     - For variables with a lot of missing values (> 2/3 of subjects), the first measurement is imputed with the median of a reference population (i.e. Geneva stroke unit)
        - Logs: labels_imputed_from_reference_population.csv
     - Path: ./handling_missing_values/impute_missing_values.py

9. Normalisation
    - For continuous variables (Quartiles/mean/sd stats are from reference population):
       - Winsorize values outside the upper and lower bounds of 1⋅5 times the IQR are set to the upper and lower limits of the range
       - Scale to a mean of 0 with an SD of 1
    - Path: ./normalisation/normalisation.py
    

### B. Outcome preprocessing

10. Outcome preprocessing
   - Path: ./outcome_preprocessing/outcome_preprocessing.py

### C. Verify preprocessing

11. Verify preprocessing
   - Path: ./preprocessing_verification/
   - Verify that the preprocessing steps were applied correctly
   - Verify that the outcome preprocessing was applied correctly

### D. Visualise preprocessed data

12. Visualise preprocessed data
   - Path: ../data_visualisation/preprocessed_patient_visualisation.ipynb


## Complete pipeline

Path: ./preprocessing_pipeline

Manual checks:
- Check the following file: verify_monitoring_too_distant_from_lab.csv
- Check visual representations
