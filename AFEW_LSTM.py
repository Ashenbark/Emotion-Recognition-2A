import os
import numpy as np

from Callbacks import save_best, save_val_best, stopping
from Generator import DataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Softmax
from tensorflow.keras.optimizers import Adam

# Constants
max_frame = 141
img_shape = 128 * 128
source_train = "Train_AFEW/AlignedFaces_LBPTOP_Points/AlignedFaces_LBPTOP_Points"
source_val = "Train_AFEW/AlignedFaces_LBPTOP_Points_Val/AlignedFaces_LBPTOP_Points_Val"

label_dict = {
    'Angry': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Neutral': 4,
    'Sad': 5,
    'Surprise': 6
}

names = []
for label in os.listdir(source_train):
    for video in os.listdir(source_train + '/' + label):
        names.append(f'{label}/{video}')

np.random.shuffle(names)

names_val = []
for label in os.listdir(source_val):
    for video in os.listdir(source_val + '/' + label):
        names_val.append(f'{label}/{video}')

np.random.shuffle(names_val)

model = Sequential([
    Input((img_shape, max_frame)),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(32),
    Dense(7),
    Softmax()
])

model.summary()
opt = Adam(learning_rate=0.01)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

training_data = DataGenerator(source_train, names, label_dict, 16, img_shape, max_frame, shuffle=False)
test_data = DataGenerator(source_val, names_val, label_dict, 16, img_shape, max_frame)
history = model.fit(training_data, validation_data=test_data, verbose=1, epochs=300,
                    callbacks=[save_best, save_val_best, stopping])

np.save('history.npy', history.history)
