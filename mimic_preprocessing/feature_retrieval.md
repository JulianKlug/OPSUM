
# Lab values

Should all be in lab events

1. [ ] sodium
2. [ ] chlore
3. [ ] hemoglobine glyquee
4. [ ] ALAT
5. [ ] calcium corrige
6. [ ] triglycerides
7. [ ] fibrinogene
8. [ ] INR
9. [ ] proBNP
10. [ ] PTT
11. [ ] potassium
12. [ ] bilirubine totale
13. [ ] lactate
14. [ ] creatinine
15. [ ] cholesterol HDL
16. [ ] LDL cholesterol calcule
17. [ ] cholesterol total
18. [ ] proteine C-reactive
19. [ ] hemoglobine
20. [ ] glucose
21. [ ] erythrocytes
22. [ ] uree
23. [ ] leucocytes
24. [ ] ASAT
25. [ ] hematocrite
26. [ ] thrombocytes

# Monitoring

Except NIHSS, should be readily available

1. [ ] NIHSS
   - Retrieval from individual components 
   - Static already extracted: https://www.physionet.org/content/stroke-scale-mimic-iii/1.0.0/
2. [ ] Glasgow Coma Scale
3. [ ] FIO2
4. [ ] oxygen_saturation
5. [ ] systolic_blood_pressure
6. [ ] diastolic_blood_pressure
7. [ ] mean_blood_pressure
8. [ ] heart_rate
9. [ ] respiratory_rate
10. [ ] temperature
11. [ ] weight


# Admission variables

1. [x] age
2. [x] Sex
3. [x] Referral
   - Table admission, ADMISSION_LOCATION

### Prestroke condition 

4. [ ] Prestroke disability (Rankin)
 - static depending on referral
 - from admission note

### Admission medication

-> Admission note?

Ref: 
- https://github.com/MIT-LCP/mimic-code/issues/729
- https://github.com/mghassem/medicationCategories

6. [ ] Antihypert. drugs pre-stroke
6. [ ] Lipid lowering drugs pre-stroke
7. [ ] Antiplatelet drugs
8. [ ] Anticoagulants

### Comorbidities 

Sources:
- CHARTEVENTS: ITEMID values (225059 - "Past medical history" and 225811 - "CV - past medical history")
-Admission note

Ref: 
- https://github.com/MIT-LCP/mimic-code/issues/729
- https://opendata.stackexchange.com/questions/9098/past-medical-medication-history-in-mimic-iii

9. [ ] MedHist Hypertension
10. [ ] MedHist Diabetes
11. [ ] MedHist Hyperlipidemia
12. [ ] MedHist Smoking
13. [ ] MedHist Atrial Fibr.
14. [ ] MedHist CHD
15. [ ] MedHist PAD
16. [ ] MedHist cerebrovascular_event

### Timing variables

-> Admission note?

18. [ ] categorical_onset_to_admission_time
18. [ ] wake_up_stroke

# Procedural variables

1. [ ] categorical_IVT
   - ICD9 code: 9910
   - d_items (227056, 2666, 2667, 42575, 221319) in inputevents_mv or inputevents_cv 
2. [ ] categorical_IAT
   - Possible solution: Patients undergoing mechanical thrombectomy were identified using the ICD-9 procedure code 39.74; as previously described by Brinjikji et al.(8) We also used (MS-DRG codes 543) code for identifying patients with percutaneous mechanical thrombectomy for ischemic stroke. If a patient received ICD-9 CM code 99.10 and subsequent ICD-9 CM 00.41-00.43, 
     - Ref: doi: 10.1161/STROKEAHA.112.658781

