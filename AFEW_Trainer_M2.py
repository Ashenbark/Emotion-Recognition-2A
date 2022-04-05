from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, RNN, Input, LSTM, Softmax

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, RNN, Input
from tensorflow.keras.layers import concatenate as layer_concatenate

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
import os

import numpy as np
from SRU import SRUCell
from Generator import DataGenerator

source = "Train_AFEW\AlignedFaces_LBPTOP_Points\AlignedFaces_LBPTOP_Points"
names = []
label_dict = {
    'Angry': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Neutral': 4,
    'Sad': 5,
    'Surprise': 6
}

for label in os.listdir(source):
    for video in os.listdir(source + '/' + label):
        names.append(f'{label}/{video}')

np.random.shuffle(names)
max_frame = 141
img_shape = 128 * 128

training_data = DataGenerator(source, names, label_dict, 32, img_shape, max_frame, shuffle=False)

save_best = ModelCheckpoint(filepath="best_tri", monitor='loss', save_best_only=True,
                            save_freq='epoch', save_weights_only=True, verbose=1)
save_val_best = ModelCheckpoint(filepath="best_val_tri", monitor='val_loss', save_best_only=True,
                                save_freq='epoch', save_weights_only=True, verbose=1)

stopping = EarlyStopping(monitor="val_loss", patience=30)

input_sequence = Input((img_shape, max_frame))
input_sequence_half1 = Input((img_shape, np.floor(max_frame)))
input_sequence_half2 = Input((img_shape, np.ceil(max_frame)))

cell = SRUCell(num_stats=40, mavg_alphas=[0.0, 0.5, 0.9, 0.99, 0.999], recur_dims=10)
cell1 = SRUCell(num_stats=25, mavg_alphas=[0.0, 0.5, 0.99, 0.999], recur_dims=10)
cell2 = SRUCell(num_stats=25, mavg_alphas=[0.0, 0.5, 0.99, 0.999], recur_dims=10)

rnn = RNN([cell], return_sequences=False)(input_sequence)
rnn1 = RNN([cell1], return_sequences=False)(input_sequence_half1)
rnn2 = RNN([cell2], return_sequences=False)(input_sequence_half2)

intermediate = Dense(32, activation='relu')(layer_concatenate([rnn, rnn1, rnn2]))

output = Dense(len(label_dict), activation='softmax')(intermediate)

model = Model(inputs=[input_sequence, input_sequence_half1, input_sequence_half2], outputs=[output])

model.load_weights("best_tri")
opt = SGD(learning_rate=0.05)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(training_data, verbose=1, use_multiprocessing=False, epochs=30)
