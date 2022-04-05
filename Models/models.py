from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Softmax, RNN, Conv1D, Conv2D, GRU
import Preprocessor
import SRU

def modelLSTM(n_frame):
    model = Sequential([
        Input((n_frame, Preprocessor.img_shape)),
        LSTM(128),
        Dense(32, activation='relu'),
        Dense(7),
        Softmax()
    ])
    return model

def modelDoubleLSTM(n_frame):
    model = Sequential([
        Input((n_frame, Preprocessor.img_shape)),
        LSTM(256, return_sequences=True),
        LSTM(128),
        Dense(32, activation='relu'),
        Dense(7),
        Softmax()
    ])
    return model

def modelTest(n_frame):
    model = Sequential([
        Input((Preprocessor.img_shape, n_frame)),
        GRU(128),
        Dense(32, activation='relu'),
        Dense(7),
        Softmax()
    ])
    return model

def modelSRU(n_frame):
    inputs = Input((Preprocessor.img_shape, n_frame))
    cell = SRU.SRUCell(num_stats=50, mavg_alphas=[0.0, 0.5, 0.9, 0.99, 0.999], recur_dims=10)
    rnn = RNN([cell], return_sequences=False)(inputs)
    output = Dense(7, activation='softmax')(rnn)

    model = Model(inputs=[inputs], outputs=[output])
    return model

