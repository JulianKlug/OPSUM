import os

from prediction.utils.utils import generate_balanced_arrays

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import os
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import log_loss, roc_auc_score, matthews_corrcoef
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from prediction.outcome_prediction.data_loading.data_formatting import format_to_linear_table
from prediction.utils.scoring import precision, recall, matthews

'''
Perceptron "Aggregated history"
Reference: Nielsen AB, Thorsen-Meyer HC, Belling K, et al. Survival prediction in intensive-care units based on aggregation of long-term disease history and acute physiology: a retrospective study of the Danish National Patient Registry and electronic patient records. Lancet Digit Health. 2019;1(2):e78-e89. doi:10.1016/S2589-7500(19)30024-X
N.B: the model originally had as input min/max from the first 24h hours
'''


# define constants
n_splits = 5
n_epochs = 5000
seed = 42

# define K fold
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# define neural network function
def create_model(units:int, outcome:str, features_df_path:str, outcomes_df_path:str, output_dir:str):
    np.random.seed(seed)
    optimal_nnet = pd.DataFrame()

    # load and format data
    X, y = format_to_linear_table(features_df_path, outcomes_df_path, outcome)
    X = np.asarray(X).astype('float32')

    # split into training and independent test sets
    X_model, X_test, y_model, y_test, ix_model, ix_test = train_test_split(X, y,
                                                                           range(X.shape[0]), test_size=0.15,
                                                                           random_state=seed)

    # run CV-folds
    i=0
    # split data further into training and validation sets for each CV
    for train_idx, val_idx in kfold.split(X_model, y_model):
        X_train, y_train = X_model[train_idx], y_model[train_idx].ravel()
        X_val, y_val = X_model[val_idx], y_model[val_idx].ravel()
        i += 1

        ### MODEL ARCHITECTURE ###
        input_layer = Input(shape = (X.shape[1], ))
        if units == 0:
            output_layer = Dense(1, activation='sigmoid')(input_layer)
        else:
            hidden = Dense(units, activation='sigmoid')(input_layer)
            output_layer = Dense(1, activation='sigmoid')(hidden)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer='RMSprop',
                      metrics=['accuracy', precision, recall, matthews])

        # define checkpoint
        filepath1 = os.path.join(output_dir, f'perceptron_{outcome}_{units}_{i}.hdf5')
        checkpoint1 = ModelCheckpoint(filepath1,
                            monitor = 'val_matthews', verbose = 0, save_best_only = True, mode = 'max')

        train_hist = model.fit_generator(generate_balanced_arrays(X_train, y_train), callbacks=[checkpoint1],
                                         epochs=n_epochs, validation_data=[X_val, y_val], steps_per_epoch=1,
                                         verbose=0)

        # gather results
        train_hist_df = pd.DataFrame.from_dict(train_hist.history)

        model.load_weights(filepath1)
        model_y_test = model.predict(X_test).reshape(len(y_test))
        model_y_pred_test = np.where(model_y_test > 0.5, 1, 0)
        model_auc_test = roc_auc_score(y_test, model_y_test)
        model_mcc_test = matthews_corrcoef(y_test, model_y_pred_test)
        model_y_val = model.predict(X_val).reshape(len(y_val))
        model_y_pred_val = np.where(model_y_val > 0.5, 1, 0)
        model_auc_val = roc_auc_score(y_val, model_y_val)
        model_mcc_val = matthews_corrcoef(y_val, model_y_pred_val)
        model_y_train = model.predict(X_train).reshape(len(y_train))
        model_y_pred_train = np.where(model_y_train > 0.5, 1, 0)
        model_auc_train = roc_auc_score(y_train, model_y_train)
        model_mcc_train = matthews_corrcoef(y_train, model_y_pred_train)

        mcc_max = train_hist_df.loc[train_hist_df['val_matthews'] == train_hist_df['val_matthews'].max()].head(1)

        # save max performance:
        mcc_max['epoch'] = mcc_max.index.tolist()
        mcc_max['CV'] = i
        mcc_max['units'] = units
        mcc_max['outcome'] = outcome
        mcc_max['auc_train'] = model_auc_train
        mcc_max['auc_val'] = model_auc_val
        mcc_max['auc_test'] = model_auc_test
        mcc_max['mcc_train'] = model_mcc_train
        mcc_max['mcc_val'] = model_mcc_val
        mcc_max['mcc_test'] = model_mcc_test
        optimal_nnet = optimal_nnet.append(mcc_max)

    optimal_nnet.to_csv(os.path.join(output_dir, f'perceptron_{outcome}_{units}.tsv'), sep='\t', index=False)


def get_NN(args):
    create_model(units = args['neurons'], outcome = args['outcome'], features_df_path = args['feature_df_path'],
                 outcomes_df_path = args['outcome_df_path'], output_dir = args['output_dir'])
    print('\nDONE: {}'.format(args))


if __name__=='__main__':
    import multiprocessing
    print('multiprocessing loaded')
    ### parse output in parallel

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature_df_path', type=str, help='path to input feature dataframe')
    parser.add_argument('-p', '--outcome_df_path', type=str, help='path to outcome dataframe')
    parser.add_argument('-o', '--outcome', type=str, help='selected outcome')
    parser.add_argument('-O', '--output_dir', type=str, help='Output directory')
    cli_args = parser.parse_args()

    # make parameter dictionary
    param_dict = {}
    param_dict['neurons'] = [0, 2, 3, 5, 10, 25, 50, 100, 200, 300]
    param_dict['outcome'] = [cli_args.outcome]
    param_dict['feature_df_path'] = [cli_args.feature_df_path]
    param_dict['outcome_df_path'] = [cli_args.outcome_df_path]
    param_dict['output_dir'] = [cli_args.output_dir]

    # save parameters as json
    with open(os.path.join(cli_args.output_dir, 'perceptron_parameters.json'), 'w') as f:
        json.dump(param_dict, f)

    # create permutations
    keys, values = zip(*param_dict.items())
    all_args = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # ## run multiprocessing
    number_processes = 30
    pool = multiprocessing.Pool(number_processes)
    # # for args in all_args make nnet
    pool.map(get_NN, all_args)