from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Masking


def lstm_generator(x_time_shape, x_channels_shape, masking, n_units, activation, dropout, n_layers) -> Model:
    """
    Generates a LSTM model

    This code is adapted from an original work by Thorsen-Meyer et al.
    Reference: Thorsen-Meyer H-C, Nielsen AB, Nielsen AP, et al. Dynamic and explainable machine learning prediction of mortality in patients in the intensive care unit: a retrospective study of high-frequency data in electronic patient records. Lancet Digital Health 2020; published online March 12. https://doi.org/10.1016/ S2589-7500(20)30018-2.

    Arguments:
        x_time_shape {int} -- Time dimension of the input
        x_channels_shape {int} -- Number of channels of the input
        masking {bool} -- Whether to use masking or not
        n_units {int} -- Number of units in the LSTM layer
        activation {str} -- Activation function of the LSTM layer
        dropout {float} -- Dropout rate of the LSTM layer
        n_layers {int} -- Number of LSTM layers

    Returns:
        Model {Model}-- LSTM model
    """

    ### MODEL ARCHITECTURE ###
    n_hidden = 1
    input_layer = Input(shape=(x_time_shape, x_channels_shape))
    if masking:
        # masking layer
        masking_layer = Masking(mask_value=0.)(input_layer)
        if n_layers == 1:
            # add first LSTM layer
            lstm = LSTM(n_units, activation=activation, recurrent_dropout=dropout)(masking_layer)
        else:
            # add first LSTM layer
            lstm = LSTM(n_units, activation=activation, recurrent_dropout=dropout,
                        return_sequences=True)(masking_layer)
    else:
        if n_layers == 1:
            # add first LSTM layer
            lstm = LSTM(n_units, activation=activation, recurrent_dropout=dropout)(input_layer)
        else:
            lstm = LSTM(n_units, activation=activation, recurrent_dropout=dropout,
                        return_sequences=True)(input_layer)
    while n_hidden < n_layers:
        n_hidden += 1
        if n_hidden == n_layers:
            # add additional hidden n_layers
            lstm = LSTM(n_units, activation=activation, recurrent_dropout=dropout)(lstm)
        else:
            lstm = LSTM(n_units, activation=activation, recurrent_dropout=dropout,
                        return_sequences=True)(lstm)

    # add output layer
    output_layer = Dense(1, activation='sigmoid')(lstm)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model