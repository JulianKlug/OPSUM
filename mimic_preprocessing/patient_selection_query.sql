
WITH selection AS
(
WITH co AS
(
SELECT icu.subject_id, icu.hadm_id, icu.icustay_id, dx.icd9_code
, EXTRACT(EPOCH FROM outtime - intime)/60.0/60.0 as icu_length_of_stay_h
, EXTRACT('epoch' from icu.intime - pat.dob) / 60.0 / 60.0 / 24.0 / 365.242 as age
, RANK() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS icustay_id_order

FROM icustays icu
INNER JOIN patients pat
  ON icu.subject_id = pat.subject_id
INNER JOIN diagnoses_icd dx
  ON icu.hadm_id = dx.hadm_id
)

SELECT
  co.subject_id, co.hadm_id, co.icustay_id, co.icu_length_of_stay_h
  , co.age
  , co.icustay_id_order
  , co.icd9_code
  , CASE
    WHEN co.icu_length_of_stay_h < 12 then 1
    ELSE 0 END
    AS exclusion_los
    , CASE
        WHEN co.age < 17 then 1
    ELSE 0 END
    AS exclusion_age
    , CASE
        WHEN co.icustay_id_order != 1 THEN 1
    ELSE 0 END
    AS exclusion_first_stay
    , CASE
        WHEN co.icd9_code LIKE '433%' OR co.icd9_code LIKE '434%' OR co.icd9_code LIKE '436%'  THEN 0
    ELSE 1 END
    AS exclusion_discharge_diagnosis
FROM co
)
