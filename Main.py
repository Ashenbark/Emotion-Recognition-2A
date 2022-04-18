import numpy as np
import random
import math
import os
from tensorflow import device

import Callbacks
import Models
import Generator

source_train = "Train_AFEW/AlignedFaces_LBPTOP_Points/AlignedFaces_LBPTOP_Points"
source_val = "Train_AFEW/AlignedFaces_LBPTOP_Points_Val/AlignedFaces_LBPTOP_Points_Val"

names = []
names_val = []
label_dict = {
    'Angry': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Neutral': 4,
    'Sad': 5,
    'Surprise': 6
}

# Get the names of the videos for the generator
for label in os.listdir(source_train):
    for video in os.listdir(source_train + '/' + label):
        names.append(f'{label}/{video}')

for label in os.listdir(source_val):
    for video in os.listdir(source_val + '/' + label):
        names_val.append(f'{label}/{video}')

random.Random(2022).shuffle(names) #set a seed to replicate validation split
names_len = len(names)
train_split = 0.9
names_train = names[:math.floor(train_split*names_len)]
names_val = names[math.floor(train_split*names_len):]

# Defined dimensions
max_frame = 141
img_shape = 128 * 128
n_frame = 30

# Create generators to be used
training_data = Generator.DataGenerator(source_train, names_train, label_dict, 16, img_shape, n_frame, shuffle=True)
val_data = Generator.DataGenerator(source_train, names_val, label_dict, 16, img_shape, n_frame, shuffle=True)
test_data = Generator.DataGenerator(source_val, names_val, label_dict, 16, img_shape, n_frame, shuffle=False)

# Definition of the model used
model = Models.modelTest(n_frame)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) #, 'categorical_accuracy'])

# Loading of previously existing weights
try:
    model.load_weights("save/best_GRU")
except Exception as e:
    print(e)


# Get the number of epochs the model has already trained on
initial = 0
try:
    history = np.load("history.npy", allow_pickle=True)
    initial = len(history.item()['loss'])
    #history.close()
except Exception as e:
    print(e)

# model.evaluate(test_data, verbose=1)

history = model.fit(training_data, validation_data=val_data, epochs=100+initial, verbose=1, validation_freq=1,
                    initial_epoch=initial,
                    callbacks=[Callbacks.save_best, Callbacks.save_val_best])#, Callbacks.stopping])

np.save("history.npy", history.history)
