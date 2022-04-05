import math
import os

from tensorflow import keras
import cv2
import numpy as np

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, source, names, label_dict, batch_size=16, dim=128 * 128, n_frames=141,
                 n_classes=7, shuffle=False):
        'Initialization'
        self.source = source
        self.names = names
        self.label_dict = label_dict
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
            y[i,:] = self.label_dict[(name.split('/')[0])]

            j = 0
            frames = os.listdir(self.source + '/' + name)

            if "desktop.ini" in frames:
                frames.remove("desktop.ini")

            for frame in frames:
                # Store sample
                img = cv2.imread(self.source + '/' + name + '/' + frame, cv2.IMREAD_GRAYSCALE)

                img = img/255.
                img = img.flatten()
                X[i, :, j] = img
                j += 1

            while j < self.n_frames:
                X[i, :, j] = np.zeros(self.dim)
                j += 1

        #print(X.shape, y.shape)
        return X, y

class DataGenerator2(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, source, names, label_dict, batch_size=16, dim=128 * 128, n_frames=141,
                 n_classes=7, shuffle=False):
        'Initialization'
        self.source = source
        self.names = names
        self.label_dict = label_dict
        self.dim = dim
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.ntemptest = []
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.names) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Find list of names
        names_temp = self.names[index * self.batch_size:(index + 1) * self.batch_size]
        self.ntemptest = names_temp

        # Generate data
        X, y = self.__data_generation(names_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # print('test')
        if self.shuffle == True:
            np.random.shuffle(self.names)

    def __data_generation(self, names_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_frames)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_frames), dtype=np.float32)
        y = np.zeros((self.batch_size, self.n_classes), dtype='uint8')

        # Generate data
        for i, name in enumerate(names_temp):
            y[i, self.label_dict[(name.split('/')[0])]] = 1

            j = 0
            frames = os.listdir(self.source + '/' + name)

            if "desktop.ini" in frames:
                frames.remove("desktop.ini")

            stock_frames = []
            for frame in frames:
                # Store sample
                img = cv2.imread(self.source + '/' + name + '/' + frame, cv2.IMREAD_GRAYSCALE)

                img = img / 255.
                img = img.flatten()
                stock_frames.append(img)

                # X[i, :, j] = img
                j += 1

            for k in range(self.n_frames):
                X[i, :, k] = stock_frames[math.floor(k/self.n_frames)]

        # print(X.shape, y.shape)
        assert not np.any(np.isnan(X))
        # quit()
        return X, y

# def getOneImage(source, n_frames):
#
#
#     for i, name in enumerate(names_temp):
#         y[i, :] = label_dict[(name.split('/')[0])]
#
#         j = 0
#         frames = os.listdir(source + '/' + name)
#
#         if "desktop.ini" in frames:
#             frames.remove("desktop.ini")
#
#         for frame in frames:
#             # Store sample
#             img = cv2.imread(source + '/' + name + '/' + frame, cv2.IMREAD_GRAYSCALE)
#
#             img = img / 255.
#             img = img.flatten()
#             stock_frames.append(img)
#
#             # X[i, :, j] = img
#             j += 1
#
#         for k in range(n_frames):
#             X[i, :, k] = stock_frames[math.floor(k / n_frames)]