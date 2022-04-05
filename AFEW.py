from tensorflow.keras.preprocessing.image import load_img
from time import time

import os

from SRU import SRUCell

import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, RNN, Input, LSTM

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import device
from tensorflow.keras.optimizers import Adam, SGD

max_frame = 141
img_shape = 128*128
num_classes = [0]

source = "Train_AFEW\AlignedFaces_LBPTOP_Points\AlignedFaces_LBPTOP_Points"

start = [time(), time()]
print("Starting importing data...")


def import_data(source, label, num_class):
    zeropad = np.zeros(img_shape)

    videos = os.listdir(f"{source}/{label}")
    video_list = np.empty((len(videos), max_frame, img_shape), dtype=np.float32)

    j = 0

    for video in videos:

        frames = os.listdir(f"{source}/{label}/{video}")
        k = 0

        if "desktop.ini" in frames:
            frames.remove("desktop.ini")

        for frame in frames:
            img = load_img(f"{source}/{label}/{video}/{frame}", color_mode="grayscale")
            img = np.asarray(img, dtype=np.float32)
            img /= 255.
            img = img.flatten()

            video_list[j, k, :] = img
            k += 1

        # Zero-padding
        while video_list.shape[1] < max_frame:
            video_list[j, k, :] = zeropad
            k += 1

        j += 1

    label_list = np.array([num_class[0] for i in range(len(video_list))])
    num_class[0] += 1

    print(f"Label {label} has been imported. There were {len(video_list)} videos in it.")
    # print(f"It took {time()-start[1]}s")
    start[1] = time()

    return video_list, label_list


nb_vid = 0
for label in os.listdir(f"{source}"):
    nb_vid += len(os.listdir(f"{source}/{label}"))

labels = os.listdir(f"{source}")

list_arrays = []
list_labels = []

for label in labels:
    video_x, video_y = import_data(source, label, num_classes)
    list_arrays.append(video_x)
    list_labels.append(video_y)
    del video_x
    del video_y


x_train = np.concatenate(list_arrays, axis=0)
del list_arrays
y_train = np.concatenate(list_labels, axis=0)
del list_labels

print(x_train.__sizeof__() / (1024 ** 3))
print(y_train.__sizeof__() / (1024 ** 1))
# print(x_train.dtype)

x_train.swapaxes(1, 2)
#x_train = np.expand_dims(x_train, -1)
y_train = np.expand_dims(y_train, -1)

print(f"Finished importing data. It took {time() - start[0]}s")


print(x_train.shape)
print(y_train.shape)

# opt_adam = tf.keras.optimizers.Adam(learning_rate=0.00025)
save_best = ModelCheckpoint(filepath="best", monitor='loss', save_best_only=True,
                            save_freq='epoch', save_weights_only=True, verbose=1)
save_val_best = ModelCheckpoint(filepath="best_val", monitor='val_loss', save_best_only=True,
                                save_freq='epoch', save_weights_only=True, verbose=1)

print(x_train.shape[1:])
# inputs = Input(x_train.shape[1:])
# cell = SRUCell(num_stats=50, mavg_alphas=[0.0, 0.5, 0.9, 0.99, 0.999], recur_dims=10)
# rnn = RNN([cell], return_sequences=False)(inputs)
# output = Dense(num_classes[0], activation='softmax')(rnn)
#
# model = Model(inputs=[inputs], outputs=[output])

model = Sequential([
    Input((Preprocessor.img_shape, n_frame)),
    LSTM(128),
    Dense(32),
    Dense(7),
    Softmax()
])

#model.load_weights("best_val_tri")
opt = SGD(learning_rate=0.05)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

x_train, y_train = unison_shuffled_copies(x_train, y_train)
print(y_train.shape)
quit()

with device("/cpu:0"):
    history = model.fit(x_train, y_train, batch_size=68, epochs=300, verbose=1,
                        validation_split=0.1, callbacks=[save_best, save_val_best])
input()
