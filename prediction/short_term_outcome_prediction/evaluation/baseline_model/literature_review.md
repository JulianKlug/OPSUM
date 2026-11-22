# Admission predictors of early neurological deterioration (END): literature summary

Purpose: choose the literature-based variable set of the admission-only baseline model (reviewer response).
Scope: systematic reviews/meta-analyses and validated prediction scores for END after acute ischaemic stroke.

## Definitions vary
- Most studies: NIHSS increase ≥4 within 24 h (END24); others ≥2 points, up to 72 h–7 days.
- Reported incidence: 2.2–37.5% (END24, Seners 2015); 6.6–43.7% across prediction-model studies (Zheng 2026).
- Our definition: NIHSS increase ≥4 over running minimum, any time in the first 72 h, predicted 6 h ahead. Published AUCs are therefore not directly comparable.

## Evidence

| Source | Population | Consistent admission predictors |
|---|---|---|
| Shi et al. 2023, *J Integr Neurosci* 22:52 — SR/MA, 29 studies, 65,960 pts | Reperfusion-treated (IVT/EVT), END24 | Age, admission SBP, admission glucose, onset-to-treatment time, hypertension, diabetes, atrial fibrillation, ICA occlusion |
| Seners et al. 2015, *JNNP* 86:87–94 — SR | < 8 h from onset, END24 | No recanalisation/reocclusion, large infarct, ICH (most consistent); admission: hyperglycaemia, no prior aspirin, prior TIA, proximal occlusion, early CT changes |
| Seners & Baron 2018, *J Neurol* — review | Unexplained END | Proximal occlusion, infarct growth into penumbra, haemodynamic/thrombotic mechanisms |
| Zheng et al. 2026, *Front Neurol* (10.3389/fneur.2026.1737871) — SR of 45 models / 23 studies | All AIS | Baseline NIHSS and SBP most frequently retained; all models high risk of bias (PROBAST) |
| Girot et al. 2020, *Stroke* 51:2943 | EVT, unexplained END | Diabetes, prestroke mRS ≥2, general anaesthesia, *lower* admission NIHSS |
| Bourcier et al. 2023 (HERMES post hoc) | EVT, unexplained END | Lower NIHSS, higher glucose, poor collaterals |
| Seners et al. 2019, *JAMA Neurol* | Minor stroke + LVO, IVT alone | Occlusion site, thrombus length (C = 0.76) |

## Published discrimination of admission-based models
- EVT nomograms: C 0.65–0.68 (derivation/validation).
- IVT scores: up to C 0.85 (CNS-LAND, END24 or death), mostly single-centre, high risk of bias.
- General AIS nomogram (labs + SBP): AUROC 0.72 internal, 0.67–0.69 external.

## Mapping to available admission variables

| Literature predictor | Model feature(s) | Note |
|---|---|---|
| Age | `age` | |
| Admission NIHSS | `median_NIHSS` (hour 0) | hour-0 value = registry admission NIHSS |
| Glucose | `glucose` | |
| SBP | `median_systolic_blood_pressure` | |
| Hypertension, diabetes, AF | `medhist_hypertension_yes`, `medhist_diabetes_yes`, `medhist_atrial_fibr._yes` | |
| Prior TIA/stroke | `medhist_cerebrovascular_event_true` | |
| Prior antiplatelet | `antiplatelet_drugs_yes` | |
| Onset-to-treatment | `categorical_onset_to_admission_time_*` | onset-to-admission is the available proxy |
| Reperfusion therapy | `categorical_ivt_*`, `categorical_iat_*` | |
| Proximal / large-vessel occlusion | `vascular_occlusion` | ICA vs other sites not distinguished |
| Infarct (core) size | `cbf_lt_30` | ASPECTS, early CT changes, collaterals, thrombus length not available |

Not represented: recanalisation/reocclusion and ICH (post-baseline events), collaterals, ASPECTS, general anaesthesia.
Imaging labels are unavailable in MIMIC (population-imputed), so they carry no information in external validation.

## Sources
- [Shi et al. 2023, J Integr Neurosci](https://doi.org/10.31083/j.jin2202052)
- [Seners et al. 2015, JNNP (PMID 24970907)](https://www.pubmed.gov/24970907/)
- [Seners & Baron 2018, J Neurol](https://link.springer.com/content/pdf/10.1007/s00415-017-8490-3.pdf)
- [Zheng et al. 2026, Front Neurol](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2026.1737871/full)
- [Girot et al. 2020, Stroke](https://doi.org/10.1161/STROKEAHA.120.029494)
- [CNS-LAND score](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10552779/)
- [EVT END nomogram](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10548384/)
- [AIS END nomogram, Front Neurol 2024](https://www.frontiersin.org/articles/10.3389/fneur.2024.1433010/full)

Items taken from abstracts/snippets; verify effect sizes against full texts before citing.
