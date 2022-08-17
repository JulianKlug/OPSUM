# MIMIC

Gist: use MIMIC as additional dataset

Versions
- MIMIC IV: 2011-2019
  - Advantage: more recent data, more similar to GVA dataset
- MIMIC III: 2001-2012
  - Advantage: stroke admissions already extracted (https://www.physionet.org/content/stroke-scale-mimic-iii/1.0.0/)


## MIMIC IV preprocessing

Outline
- Patient selection: use MIMIC IV ED admission diagnosis (https://mimic.mit.edu/docs/iv/modules/ed/diagnosis/) to select stroke patients
  - TODO: verify this works for hospitalised patients
  - Alternative: identify with discharge diagnosis

- Extract patient data 
