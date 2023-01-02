# Data sources

For the Geneva Stroke Unit (GSU) project, we have use two main sources of data:
1. The Swiss Stroke Unit Registry
2. The Geneva EHR (DPI)

## Swiss Stroke Unit Registry
Patients: all patients seen by the geneva stroke unit team 

Date of extraction: 21.06.2022

Years: [2018-2021]

All data was reviewed by the investigators. In cases of inconsistencies, the original patient records were reviewed and post-hoc modifications were applied to the database (./stroke_registry_post_hoc_modifications).

## Geneva EHR (DPI)
Patients: all patients present in the Swiss Stroke Unit Registry

Dates of extraction:
- 20220815 (all data)
- 20221117 (general consent data)

Notes: 
- Response to consentement general available from extraction of 20221117
- However, Extraction_20221117 is missing some patients, therefore for all other variables the extraction prior is used for now (Extraction_20220815)