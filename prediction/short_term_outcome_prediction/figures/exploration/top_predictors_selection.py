import pandas as pd
import pickle
import numpy as np
import torch as ch
import os
import seaborn as sns
import matplotlib.pyplot as plt
from prediction.utils.visualisation_helper_functions import hex_to_rgb_color, create_palette
from colormath.color_objects import LabColor

shap_values_path = '/Users/jk1/temp/opsum_end/testing/with_imaging/xgb_test_results/shap_explanations_over_time/tree_explainer_shap_values_over_ts.pkl'
test_data_path = '/Users/jk1/temp/opsum_end/preprocessing/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/splits/test_data_early_neurological_deterioration_ts0.8_rs42_ns5.pth'
cat_encoding_path = '/Users/jk1/temp/opsum_end/preprocessing/with_imaging/gsu_Extraction_20220815_prepro_30012026_154047/logs_30012026_154047/categorical_variable_encoding.csv'

# load the shap values
with open(os.path.join(shap_values_path), 'rb') as handle:
    original_shap_values = pickle.load(handle)

shap_values = [np.array([original_shap_values[i] for i in range(len(original_shap_values))]).swapaxes(0, 1)][0]

X_test, y_test= ch.load(test_data_path)

features = X_test[0, 0, :, 2]

# Toggle these to match the model that produced the SHAP values
add_lag_features = True
add_rolling_features = True

# Build aggregated feature names matching aggregate_features_over_time output order:
# [features, avg_features, min_features, max_features, std_features, diff_features, timestep_feature] [lag2, lag3] [roll_mean, roll_std, roll_trend]
# features, avg_, min_, max_, std_, diff_, timestep_idx, [lag2_, lag3_], [rolling_mean_, rolling_std_, rolling_trend_]
aggregated_feature_names = list(features)
for prefix in ['avg_', 'min_', 'max_', 'std_', 'diff_']:
    aggregated_feature_names += [f'{prefix}{f}' for f in features]
aggregated_feature_names += ['timestep_idx']

if add_lag_features:
    for prefix in ['lag2_', 'lag3_']:
        aggregated_feature_names += [f'{prefix}{f}' for f in features]

if add_rolling_features:
    for prefix in ['rolling_mean_', 'rolling_std_', 'rolling_trend_']:
        aggregated_feature_names += [f'{prefix}{f}' for f in features]

aggregated_feature_names += ['base_value']
print(f'{len(aggregated_feature_names)} feature names (including base_value), SHAP columns: {shap_values.shape[2]}')

sum_over_all_shap_values = np.abs(shap_values).sum(axis=(0,1))


temp_df = pd.DataFrame({'feature': aggregated_feature_names, 'shap_value': sum_over_all_shap_values})
# remove timestep_idx and base_value from the features
temp_df = temp_df[~temp_df.feature.isin(['timestep_idx', 'base_value'])]
# remove avg_, min_, max_, std_, diff_, timestep_idx, [lag2_, lag3_], [rolling_mean_, rolling_std_, rolling_trend_] from the feature names to get the original feature names
prefixes = ['rolling_mean_', 'rolling_std_', 'rolling_trend_', 'avg_', 'min_', 'max_', 'std_', 'diff_', 'lag2_', 'lag3_',]
for prefix in prefixes:
    temp_df.loc[temp_df.feature.str.contains(prefix), 'feature'] = temp_df[temp_df.feature.str.contains(prefix)].feature.apply(lambda x: x.replace(prefix, ''))
hourly_pool_prefixes = ['median_', 'min_', 'max_']
for prefix in hourly_pool_prefixes:
    temp_df.loc[temp_df.feature.str.contains(prefix), 'feature'] = temp_df[temp_df.feature.str.contains(prefix)].feature.apply(lambda x: x.replace(prefix, ''))
blood_pressure_prefixes = ['systolic_', 'diastolic_', 'mean_']
for prefix in blood_pressure_prefixes:
    temp_df.loc[temp_df.feature.str.contains(prefix), 'feature'] = temp_df[temp_df.feature.str.contains(prefix)].feature.apply(lambda x: x.replace(prefix, ''))

# transform to absolute shap values
temp_df['absolute_shap_value'] = np.abs(temp_df['shap_value'])
# drop shap value
temp_df = temp_df.drop(columns=['shap_value'])
# sum the shap values for the same original feature names
temp_df = temp_df.groupby('feature').sum().reset_index()
temp_df.sort_values(by='absolute_shap_value', ascending=False).head(10)
top_10_features_by_mean_abs_summed_shap = temp_df.sort_values(by='absolute_shap_value', ascending=False).head(10).feature.values

print(f'Top 10 features by mean absolute summed SHAP values: {top_10_features_by_mean_abs_summed_shap}')