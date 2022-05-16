import os
import numpy as np
import pandas as pd
import itertools
import logging
from sklearn import preprocessing
from sklearn.metrics import log_loss, roc_auc_score, matthews_corrcoef
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import Callback
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from prediction.utils.scoring import precision, recall, matthews

'''
Perceptron "Aggregated history"
Reference: Thorsen-Meyer HC, Nielsen AB, Nielsen AP, et al. Dynamic and explainable machine learning prediction of mortality in patients in the intensive care unit: a retrospective study of high-frequency data in electronic patient records. Lancet Digit Health. 2020;2(4):e179-e191. doi:10.1016/S2589-7500(20)30018-2
N.B: the model originally had as input min/max from the first 24h hours
'''


# define constants
n_splits = 5
n_epochs = 5000
seed = 1234

# make parameter dictionary
param_dict={}
param_dict['neurons'] = [0,2,3,5,10,25,50,100,200,300]
param_dict['outcome'] = [0,1,2] # 0, 1, 2 represent in-hospital, 30-day and 90-day mortality, respectively

# define K fold
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
# define metrics for evaluation



# define function for balanced training
def generate_balanced_arrays(X_train, y_train):
    while True:
        positive = np.where(y_train == 1)[0].tolist()
        negative = np.random.choice(np.where(y_train == 0)[0].tolist(),
                                            size = len(positive),
                                            replace = False)
        balance = np.concatenate((positive, negative), axis = 0)
        np.random.shuffle(balance)
        input_ = X_train[balance]
        target = y_train[balance]
        yield input_, target


# define neural network function

def create_model(units, outcome):
    np.random.seed(seed)
    optimal_nnet = pd.DataFrame()

    # TODO fix data loading
    # load data
    epr_data = pd.read_table( '< scaled_data.tsv >')
    registry_data = pd.read_table( '< registry_data.tsv >', sep = '\t')  # gather data
    merged_data = pd.merge(epr_data, registry_data)
    # split data in input and labels
    data_np = merged_data.values
    input_cols = data_np.shape[1]-3
    X = data_np[:, 1:input_cols]
    y = data_np[:, input_cols + outcome].astype(int) # select outcome

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

        model.compile(loss='binary_crossentropy', optimizer='RSMprop',
                      metrics=['accuracy', precision, recall, matthews])

        # define checkpoint
        # TODO fix path
        filepath1='////<PATH>{}{}{}.hdf5'.format(units, outcome,i)
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

    # TODO fix path
    optimal_nnet.to_csv('//////<PATH>{}{}.tsv'.format(units, outcome), sep='\t', index=False)


if __name__=='__main__':
    import multiprocessing
    print('multiprocessing loaded')
    ### parse output in parallel
    allNames = sorted(param_dict)
    all_args = [item for item in itertools.product(*(param_dict[Name] for Name in allNames))]
    ### for args in all_args make nnet
    def get_NN(args):
        neurons, outcome = args
        create_model(units=neurons, outcome=outcome)
        print('\nDONE: {}'.format(args))

    ### run multiprocessing
    number_processes = 30
    pool = multiprocessing.Pool(number_processes)
    pool.map(get_NN, all_args)