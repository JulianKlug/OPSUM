
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
    score = 0

    if age > 75:
        score += 1
    if admission_NIHSS > 18:
        score +=1
    # Units: mmol/L
    if admission_glucose > 8.4:
        score += 1

    # Return probability of good outcome: mRs <4 on hospital discharge
    if score == 0:
        return 1- 0.44
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
    score = age + admission_NIHSS
    if treated:
        if score < 100:
            return 0.554
        else:
            return 0.056
    else: # not treated patients
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
    if hist_afib:
        chronic_disease_scale +=1
    if hist_hypertension:
        chronic_disease_scale += 1
    if hist_diabetes:
        chronic_disease_scale += 1

    score += chronic_disease_scale

    if score < 3:
        return 0.647
    elif score < 6:
        return 0.435
    else:
        return 0.106

