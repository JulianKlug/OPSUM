# High frequency stroke unit data preprocessing

1. Apply post-hoc modifications to stroke registry data
   - Path: ./stroke_registry_post_hoc_modifications/stroke_registry_post_hoc_modifications.ipynb
   - Correct any mistakes in stroke registry file

2. Select patients from stroke registry data
   - Path: ./patient_selection/patient_selection.ipynb
   - Inclusion criteria: > 17y, ischemic stroke, inpatient/non-transferred, not refusing to participate

3. Select variables
   - Path: ./variable_assembly/selected_variables.xlsx

4. Preprocessing & database creation
   - Path: ./variable_assembly/variable_database_assembly.py
   - Preprocessing involves restriction to plausible ranges: ./possible_ranges_for_variables.xlsx

5. Transform timestamps to relative timestamps from first measure
   - Path: ./variable_assembly/relative_timestamps.py

6. Restrict to desired time range 
   - Chosen time range: 72h
   - Path: ./variable_assembly/relative_timestamps.py

7. Normalisation
   - For continuous variables:
      - Winsorize values outside the upper and lower bounds of 1⋅5 times the IQR are set to the upper and lower limits of the range
      - Scale to a mean of 0 with an SD of 1
   - Path: ./normalisation/normalisation.py

TODO:
- complete normalisation
- Partition to hourly values
- Fill missing values