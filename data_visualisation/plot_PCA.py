import pandas as pd
import numpy as np
import seaborn as sns
from ppca import PPCA
import matplotlib.pyplot as plt

save_path = '/Users/jk1/temp/opsum_PCA_test'
#%%
data_path = '/Users/jk1/stroke_datasets/stroke_unit_dataset/per_value/Extraction_20211110'
admission_data_path = '/Users/jk1/OneDrive - unige.ch/stroke_research/geneva_stroke_unit_dataset/data/stroke_registry/post_hoc_modified/stroke_registry_post_hoc_modified.xlsx'
patient_selection_path = '/Users/jk1/temp/opsum_extration_output/high_frequency_data_patient_selection.csv'
#%%
from preprocessing.variable_assembly.variable_database_assembly import assemble_variable_database

feature_df = assemble_variable_database(data_path, admission_data_path, patient_selection_path)
#%%
from preprocessing.variable_assembly.relative_timestamps import transform_to_relative_timestamps

restricted_feature_df = transform_to_relative_timestamps(feature_df, drop_old_columns=False, restrict_to_time_range=True)
#%%
from preprocessing.normalisation.normalisation import normalise_data

normalised_restricted_feature_df = normalise_data(restricted_feature_df, verbose=True)
#%%
#%%
normalised_restricted_feature_df['relative_sample_date_hourly_cat'] = np.floor(normalised_restricted_feature_df['relative_sample_date'])
#%%
from preprocessing.outcome_preprocessing.outcome_preprocessing import preprocess_outcomes

stroke_registry_df = pd.read_excel(admission_data_path)
patient_selection_df = pd.read_csv(patient_selection_path, dtype=str)
outcome_df = preprocess_outcomes(stroke_registry_df, patient_selection_df)

# fit PCA on hour0
hour0_df = normalised_restricted_feature_df[normalised_restricted_feature_df['relative_sample_date_hourly_cat'] == 0]
#%%
# for simplicity dropping variables with duplicated values (but ideally duplicates should be replacing by median / mode /min / max in preprocessing)
hour0_df = hour0_df.drop_duplicates(['case_admission_id', 'sample_label'])
hour0_df = hour0_df[['case_admission_id', 'sample_label', 'value']].pivot(index='case_admission_id', columns='sample_label', values='value')
#%%
hour0_df = hour0_df.reset_index()
#%%
#%%
#%%
hour0_df_with_outcomes = pd.merge(hour0_df, outcome_df, left_on='case_admission_id', right_on='patient_admission_id')
#%%
columns_to_drop = ['case_admission_id'] + outcome_df.columns.tolist()
input_hour0_df = hour0_df_with_outcomes.drop(columns_to_drop, axis=1)
#%%
 # factorize columns if it contains strings
factorized_hour0_df = input_hour0_df.apply(lambda x: pd.factorize(x)[0] if type(x.mode(dropna=True)[0]) == str else x)
#%%
#%%
ppca = PPCA()
#%%
factorized_hour0_df = factorized_hour0_df.astype(float)
#%%
#%%
n_components = 2
#%%
model = ppca.fit(data=factorized_hour0_df.to_numpy(), d=n_components, verbose=True)
model_hour0_params = ppca.C

#%%
# for 0 to 72 hours
for hour_bin in range(0, 72):
    hourX_df = normalised_restricted_feature_df[normalised_restricted_feature_df['relative_sample_date_hourly_cat'] == hour_bin]
    #%%
    # for simplicity dropping variables with duplicated values (but ideally duplicates should be replacing by median / mode /min / max in preprocessing)
    hourX_df = hourX_df.drop_duplicates(['case_admission_id', 'sample_label'])
    hourX_df = hourX_df[['case_admission_id', 'sample_label', 'value']].pivot(index='case_admission_id', columns='sample_label', values='value')
    #%%
    hourX_df = hourX_df.reset_index()
    #%%
    #%%
    #%%
    hourX_df_with_outcomes = pd.merge(hourX_df, outcome_df, left_on='case_admission_id', right_on='patient_admission_id')
    #%%
    columns_to_drop = ['case_admission_id'] + outcome_df.columns.tolist()
    input_hourX_df = hourX_df_with_outcomes.drop(columns_to_drop, axis=1)
    #%%
     # factorize columns if it contains strings
    factorized_hourX_df = input_hourX_df.apply(lambda x: pd.factorize(x)[0] if type(x.mode(dropna=True)[0]) == str else x)
    #%%
    #%%
    ppca = PPCA()
    #%%
    factorized_hourX_df = factorized_hourX_df.astype(float)
    #%%
    #%%
    n_components = 2
    #%%
    model = ppca.fit(data=factorized_hourX_df.to_numpy(), d=n_components, verbose=True)

    # %%
    ppca.C = model_hour0_params
    #%%
    component_matrix = ppca.transform()
    #%%
    components_with_outcomes = pd.concat([pd.DataFrame(component_matrix), hourX_df_with_outcomes['3M mRS']], axis=1)
    #%%

    # plot scatter plot of first two principal components without legend
    ax = sns.scatterplot(x=components_with_outcomes[0], y=components_with_outcomes[1], hue=components_with_outcomes['3M mRS'], legend=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    # set title with two digit hours
    ax.set_title('Hour ' + str(hour_bin).zfill(2))
    # save plot
    plt.savefig(save_path + '/PPCA_' + str(hour_bin) + '.png')
    plt.close()
    #%%
 