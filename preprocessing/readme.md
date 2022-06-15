# High frequency stroke unit data preprocessing

1. Apply post-hoc modifications to stroke registry data
   - Path: ./stroke_registry_post_hoc_modifications/stroke_registry_post_hoc_modifications.ipynb
   - Correct any mistakes in stroke registry file

   
2. Select patients from stroke registry data
   - Path: ./patient_selection/patient_selection.ipynb
   - Inclusion criteria: > 17y, ischemic stroke, inpatient/non-transferred, not refusing to participate
   - Exclusion criteria: < 12h of hospitalisation



### A. Feature preprocessing

3. Select variables
   - Path: ./variable_assembly/selected_variables.xlsx

4. Preprocessing & database creation
   - Path: ./variable_assembly/variable_database_assembly.py
   - Details: 
     - Restrict to patient selection (& filter out patients with no EHR data or EHR data with wrong dates)
       - case_admissions to those sampled within bounds of event [event date -10 days, event/admission date + 7 days]
       - Path: ./patient_selection/restrict_to_patient_selection.py
     - Preprocess EHR and stroke registry variables
       - Preprocessing involves restriction to plausible ranges: ./possible_ranges_for_variables.xlsx
     - Restrict to variable selection
     - Assemble database from lab/scales/ventilation/vitals + stroke registry subparts
   - Missing patients (not found in intersection EHR / registry) are logged under: 'missing_cids_from_feature_database.tsv' 

5. Transform timestamps to relative timestamps from first measure
   - Path: ./variable_assembly/relative_timestamps.py

6. Exclude patients with data sampled in a time window < 12h
   - Excluded patients are logged under: 'excluded_patients_with_sampling_range_too_short.tsv' 
   - Path: ./variable_assembly/relative_timestamps.py

8. Restrict to desired time range 
   - Chosen time range: 72h
   - Path: ./variable_assembly/relative_timestamps.py

9. Encoding categorical variables
   - Categorical variables are one hot encoded 
   - Path: ./encoding_categorical_variables/encoding_categorical_variables.py
   
10. Resampling to selected frequency: _hourly_
    - Downsampling selected features to hourly median/max/min values: ['NIHSS', 'oxygen_saturation', 'systolic_blood_pressure', 'diastolic_blood_pressure', 'mean_blood_pressure', 'heart_rate', 'respiratory_rate']
    - All other features are downsampled to hourly median
    - Path: ./resample_to_time_bins/resample_to_hourly_features.py

11. Imputation of missing values
     - Missing values are imputed by last observation carried forward (LOCF). 
     - Population medians in the datasets are used for missing values occurring before the first actual measurement.
     - Path: ./handling_missing_values/impute_missing_values.py

12. Normalisation
    - For continuous variables:
       - Winsorize values outside the upper and lower bounds of 1⋅5 times the IQR are set to the upper and lower limits of the range
       - Scale to a mean of 0 with an SD of 1
    - Path: ./normalisation/normalisation.py
    
### B. Outcome preprocessing

12. Outcome preprocessing
   - Path: ./outcome_preprocessing/outcome_preprocesing.py


## Complete pipeline

Path: ./preprocessing_pipeline