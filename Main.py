import numpy as np
import os

import Preprocessor
import Callbacks
import Models
import Generator

source_train = "Train_AFEW/AlignedFaces_LBPTOP_Points/AlignedFaces_LBPTOP_Points"
source_val = "Train_AFEW/AlignedFaces_LBPTOP_Points_Val/AlignedFaces_LBPTOP_Points_Val"

#x_train, y_train = Preprocessor.importDataFromSourceSingle(source_train)
#x_test, y_test = Preprocessor.importDataFromSourceSingle(source_val)

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

for label in os.listdir(source_train):
    for video in os.listdir(source_train + '/' + label):
        names.append(f'{label}/{video}')

for label in os.listdir(source_val):
    for video in os.listdir(source_val + '/' + label):
        names_val.append(f'{label}/{video}')

np.random.shuffle(names)
max_frame = 141
img_shape = 128 * 128
n_frame = 30

training_data = Generator.DataGenerator2(source_train, names, label_dict, 16, img_shape, n_frame, shuffle=True)
test_data = Generator.DataGenerator2(source_val, names_val, label_dict, 16, img_shape, n_frame, shuffle=False)

model = Models.modelTest(n_frame)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])
# model.fit(training_data, epochs=30, verbose=1)

# r = model.predict(training_data, 1)
# print(r)

try:
    model.load_weights("best_LSTM")
except Exception as e:
    print(e)


# model.evaluate(x_test, y_test, batch_size=16, verbose=1)
#
history = model.fit(training_data, validation_data=test_data, epochs=100,
                    verbose=1, callbacks=[Callbacks.save_best, Callbacks.save_val_best, Callbacks.stopping])
#
np.save('history.npy', history.history)