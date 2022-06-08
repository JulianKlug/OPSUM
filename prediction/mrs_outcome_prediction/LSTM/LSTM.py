from keras.models import Model
from keras.layers import Dense, LSTM, Input, Masking


def lstm_generator(x_time_shape, x_channels_shape, masking, n_units, activation, dropout, n_layers) -> Model:
    ### MODEL ARCHITECTURE ###
    n_hidden = 1
    input_layer = Input(shape=(x_time_shape, x_channels_shape))
    # TODO: try 1D convolutional layer without reducing time dimension too much
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

    # TODO: try adding more dense layers here
    # add output layer
    output_layer = Dense(1, activation='sigmoid')(lstm)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model