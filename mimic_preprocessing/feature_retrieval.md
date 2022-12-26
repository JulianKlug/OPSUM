
# Lab values

Should all be in lab events

1. [x] sodium
2. [x] chlore
3. [x] hemoglobine glyquee
4. [x] ALAT
5. [x] calcium corrige
- extracted from calcium and albumin
6. [x] triglycerides
7. [x] fibrinogene
8. [x] INR
9. [x] proBNP
10. [x] PTT
11. [x] potassium
12. [x] bilirubine totale
13. [x] lactate
14. [x] creatinine
15. [x] cholesterol HDL
16. [x] LDL cholesterol calcule
17. [x] cholesterol total
18. [x] proteine C-reactive
19. [x] hemoglobine
20. [x] glucose
21. [x] erythrocytes
22. [x] uree
23. [x] leucocytes
24. [x] ASAT
25. [x] hematocrite
26. [x] thrombocytes
27. [x] lactate

# Monitoring

Except NIHSS, should be readily available

1. [x] NIHSS
   - Retrieval from individual components 
   - Static already extracted: https://www.physionet.org/content/stroke-scale-mimic-iii/1.0.0/
2. [x] Glasgow Coma Scale
3. [x] FIO2
4. [x] oxygen_saturation
5. [x] systolic_blood_pressure
6. [x] diastolic_blood_pressure
7. [x] mean_blood_pressure
8. [x] heart_rate
9. [x] respiratory_rate
10. [x] temperature
11. [x] weight
12. [x] glycemia


# Admission variables

1. [x] age
2. [x] Sex
3. [x] Referral
   - Table admission, ADMISSION_LOCATION

### Prestroke condition 

4. [x] Prestroke disability (Rankin)
 - from admission note

### Admission medication

-> Admission note?

Ref: 
- https://github.com/MIT-LCP/mimic-code/issues/729
- https://github.com/mghassem/medicationCategories

6. [x] Antihypert. drugs pre-stroke
6. [x] Lipid lowering drugs pre-stroke
7. [x] Antiplatelet drugs
8. [x] Anticoagulants

### Comorbidities 

Sources:
- CHARTEVENTS: ITEMID values (225059 - "Past medical history" and 225811 - "CV - past medical history" & 'Tobacco Use History')
-Admission note

Ref: 
- https://github.com/MIT-LCP/mimic-code/issues/729
- https://opendata.stackexchange.com/questions/9098/past-medical-medication-history-in-mimic-iii

9. [x] MedHist Hypertension
10. [x] MedHist Diabetes
11. [x] MedHist Hyperlipidemia
12. [x] MedHist Smoking
13. [x] MedHist Atrial Fibr.
14. [x] MedHist CHD
15. [x] MedHist PAD
16. [x] MedHist cerebrovascular_event

### Timing variables

-> Admission note?

18. [x] categorical_onset_to_admission_time
18. [x] wake_up_stroke

# Procedural variables

--> Admission note

1. [x] categorical_IVT
   - ICD9 code: 9910
   - d_items (227056, 2666, 2667, 42575, 221319) in inputevents_mv or inputevents_cv 
2. [x] categorical_IAT
   - Possible solution: Patients undergoing mechanical thrombectomy were identified using the ICD-9 procedure code 39.74; as previously described by Brinjikji et al.(8) We also used (MS-DRG codes 543) code for identifying patients with percutaneous mechanical thrombectomy for ischemic stroke. If a patient received ICD-9 CM code 99.10 and subsequent ICD-9 CM 00.41-00.43, 
     - Ref: doi: 10.1161/STROKEAHA.112.658781

