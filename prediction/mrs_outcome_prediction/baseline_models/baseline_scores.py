import math
import pandas as pd


def check_for_truth_equivalents(input):
    '''
    Check if entry is equivalent to True equivalent value
    '''
    truth_equivalents = ['True', 'true', 'TRUE', '1', 1, True, 'yes', 'Yes', 'YES', 'y', 'Y', 't', 'T']
    if input in truth_equivalents:
        return True
    else:
        return False



def hiat_score(age, admission_NIHSS, admission_glucose):
    '''
    Houston Intra-arterial recanalization therapy Score
    Population: all patients undergoing IAT
    We scored 1 point for each variable as follows: age >75 years; NIHSS >18; and admission glucose >150 mg/dL (> 8.4 mmol/L).
    The percentage of poor outcome (defined as modified Rankin Scale score 4 to 6 on hospital discharge) by Houston IAT score was: score of 0, 44%; 1, 67%; 2, 97%; and 3, 100%
    Ref:
        H. Hallevi, A.D. Barreto, D.S. Liebeskind, et al.
        Identifying patients at high risk for poor outcome after intra-arterial therapy for acute ischemic stroke
        Stroke, 40 (2009), pp. 1780-1785
    '''
    # if any of age, admission_NIHSS or admission_glucose is nan return nan
    if pd.isnull(age) or pd.isnull(admission_NIHSS) or pd.isnull(admission_glucose):
        return math.nan

    score = 0

    if age > 75:
        score += 1
    if admission_NIHSS > 18:
        score += 1
    # Units: mmol/L
    if admission_glucose > 8.4:
        score += 1

    # Return probability of good outcome: mRs <4 on hospital discharge
    if score == 0:
        return 1 - 0.44
    if score == 1:
        return 1 - 0.67
    if score == 2:
        return 1 - 0.97
    if score == 3:
        return 1 - 1


def span100_score(age, admission_NIHSS, treated=1):
    '''
    Treated variable initially only encompasses IVT
    Population: patients treated with placebo or tPA (NINDS trial cohort)
    Returns: probability of composite favorable outcome (defined as a modified Rankin Scale score of 0 or 1, NIHSS #1, Barthel index $95, and Glasgow Outcome Scale score of 1) at 3 months
    Ref:
    G. Saposnik, A. Guzik, M. Reeves, B. Ovbiagele, S. Johnston
    Stroke prognostication using age and NIH stroke scale: SPAN-100
    Neurology, 80 (2013), pp. 21-28
    '''
    if pd.isnull(age) or pd.isnull(admission_NIHSS):
        return math.nan

    score = age + admission_NIHSS
    if treated:
        if score < 100:
            return 0.554
        else:
            return 0.056
    else:  # not treated patients
        if score < 100:
            return 0.402
        else:
            return 0.039


def thrive_score(age, admission_NIHSS, hist_hypertension, hist_diabetes, hist_afib):
    '''
    Population: patients undergoing endovascular stroke treatment
    Returns: probability Good Outcome (mRS 0–2)
    low THRIVE score of 0 –2 had good outcomes in 64.7% of cases, patients with a moderate THRIVE score of 3–5 had good outcomes in 43.5% of cases, and patients with a high THRIVE score of 6 –9 had good outcomes in 10.6% of cases
    Ref:
    A.C. Flint, B.S. Faigeles, S.P. Cullen, et al.
    THRIVE score predicts ischemic stroke outcomes and thrombolytic hemorrhage risk in VISTA
    Stroke, 44 (2013), pp. 3365-3369
    '''
    if pd.isnull(age) or pd.isnull(admission_NIHSS) or pd.isnull(hist_hypertension) or pd.isnull(hist_diabetes) or pd.isnull(hist_afib):
        return math.nan

    score = 0
    if admission_NIHSS >= 11 and admission_NIHSS < 21:
        score += 2
    if admission_NIHSS >= 21:
        score += 4
    if age >= 60 and age < 80:
        score += 1
    if age >= 80:
        score += 2

    chronic_disease_scale = 0
    if check_for_truth_equivalents(hist_afib):
        chronic_disease_scale += 1
    if check_for_truth_equivalents(hist_hypertension):
        chronic_disease_scale += 1
    if check_for_truth_equivalents(hist_diabetes):
        chronic_disease_scale += 1

    score += chronic_disease_scale

    if score < 3:
        return 0.647
    elif score < 6:
        return 0.435
    else:
        return 0.106


def thriveC_score(age, admission_NIHSS, hist_hypertension, hist_diabetes, hist_afib):
    '''
    Population: VISTA & STIS-most studies (IVT trials)
    Returns: probability Good Outcome (mRS 0–2)
    𝑃=1/(1+𝑒^−(4.94+(−0.035*𝑎𝑔𝑒)+(−0.19*𝑁𝐼𝐻𝑆𝑆)+(−0.105*𝐶𝐷𝑆1)+(−0.408*𝐶𝐷𝑆2)+(−0.702*𝐶𝐷𝑆3)))
    Ref:
    Flint AC, Rao VA, Chan SL, et al. Improved Ischemic Stroke Outcome Prediction Using Model Estimation of Outcome Probability: The THRIVE-c Calculation. International Journal of Stroke. 2015;10(6):815-821. doi:10.1111/ijs.12529
    '''
    if pd.isnull(age) or pd.isnull(admission_NIHSS) or pd.isnull(hist_hypertension) or pd.isnull(hist_diabetes) or pd.isnull(hist_afib):
        return math.nan

    chronic_disease_scale = 0
    if check_for_truth_equivalents(hist_afib):
        chronic_disease_scale += 1
    if check_for_truth_equivalents(hist_hypertension):
        chronic_disease_scale += 1
    if check_for_truth_equivalents(hist_diabetes):
        chronic_disease_scale += 1

    if chronic_disease_scale == 0:
        cds_coefficient = 0
    elif chronic_disease_scale == 1:
        cds_coefficient = -0.105
    elif chronic_disease_scale == 2:
        cds_coefficient = -0.408
    elif chronic_disease_scale == 3:
        cds_coefficient = -0.702
    else:
        raise ValueError('Chronic disease scale must be between 0 and 3')

    probability = 1 / (1 + math.exp(-(4.942
                                      + (-0.035 * age)
                                      + (-0.19 * admission_NIHSS)
                                      + cds_coefficient)))

    return probability
