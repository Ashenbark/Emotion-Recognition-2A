import os

#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow import keras
import cv2
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, RNN, Input, LSTM, Softmax

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import device
from tensorflow.keras.optimizers import Adam, SGD

from SRU import SRUCell

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


# print(names[0].split('/')[0])


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, names, batch_size=32, dim=128 * 128, n_frames=141,
                 n_classes=7, shuffle=False):
        'Initialization'
        self.names = names
        self.dim = dim
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.names) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Find list of names
        names_temp = self.names[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(names_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.names)

    def __data_generation(self, names_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_frames)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_frames), dtype=np.float32)
        y = np.empty((self.batch_size,1))

        # Generate data
        for i, name in enumerate(names_temp):
            y[i,:] = label_dict[(name.split('/')[0])]

            j = 0
            frames = os.listdir(source + '/' + name)

            if "desktop.ini" in frames:
                frames.remove("desktop.ini")

            for frame in frames:
                # Store sample
                img = cv2.imread(source + '/' + name + '/' + frame, cv2.IMREAD_GRAYSCALE)

                img = img/255.
                img = img.flatten()
                X[i, :, j] = img
                j += 1

            while j < self.n_frames:
                X[i, :, j] = np.zeros(img_shape)
                j += 1

        return X, y


training_data = DataGenerator(names, 32, img_shape, max_frame, shuffle=False)

# model = Sequential([
#     Input((img_shape, max_frame)),
#     LSTM(128),
#     Dense(7),
#     Softmax()
# ])

inputs = Input((img_shape, max_frame))
cell = SRUCell(num_stats=50, mavg_alphas=[0.0, 0.5, 0.9, 0.99, 0.999], recur_dims=10)
rnn = RNN([cell], return_sequences=False)(inputs)
output = Dense(7, activation='softmax')(rnn)

model = Model(inputs=[inputs], outputs=[output])

#opt = SGD(learning_rate=0.05)
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(training_data, verbose=1, use_multiprocessing=False, epochs=30)
