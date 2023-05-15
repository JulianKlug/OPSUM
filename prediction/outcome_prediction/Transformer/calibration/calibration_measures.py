import numpy as np
import pandas as pd
import statsmodels.api as sm


def logit(p):
    return np.log(p / (1 - p))


def cox_calibration_coefficients(y, prob):
    """
    Compute the slope and intercept of the calibration curve (Cox method)
    Gist: Fit a logistic regression model with the logit of the predicted probability as the independent variable and
            the binary outcome as the dependent variable.
    Formula: logit {P(O=1)} = a + b logit(E)
    Interpretation:
        - Perfect calibration: slope = 1, intercept = 0
        - Intercept: >0 denotes an average underestimation, and <0 denotes an average overestimation
        - Slope: slope >1 -> underestimation of high risk and overestimation of low risk
    Reference: Cox DR. Two further applications of a model for binary regression. Biometrika 1958; 45 (3–4): 592–65.

    Parameters
    :param y: 1d array of binary outcome
    :param prob: 1d array of predicted probability

    Returns
    :return: dictionary of slope and intercept
    """
    dat = pd.DataFrame({'e': prob, 'o': y})
    dat.loc[dat['e'] == 0, 'e'] = 0.0000000001
    dat.loc[dat['e'] == 1, 'e'] = 0.9999999999
    # take logit of predicted probability
    dat['logite'] = logit(dat['e'])

    X = sm.add_constant(dat['logite'])
    y = dat['o']

    model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.logit()))
    result = model.fit()

    slope = result.params[1]
    intercept = result.params[0]

    return {'slope': slope, 'intercept': intercept}
