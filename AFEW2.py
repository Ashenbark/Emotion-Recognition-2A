import random

import keras.layers
from tensorflow.keras.preprocessing.image import load_img
from time import time

import os

from SRU import SRUCell

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, RNN, Input

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import device
from tensorflow.keras.optimizers import Adam, SGD

max_frame = 141
img_shape = 128*128
num_classes = [0]

source = "Train_AFEW\AlignedFaces_LBPTOP_Points\AlignedFaces_LBPTOP_Points"

start = [time(), time()]
print("Starting importing data...")


def true_ceil(x):
    return int(np.ceil(x))


def true_floor(x):
    return int(np.floor(x))


def import_data1(source, label, num_class):

    zeropad = np.zeros(img_shape)

    videos = os.listdir(f"{source}/{label}")
    random.shuffle(videos)
    label = video.strip("/")[0]
    video_list = np.empty((len(videos), max_frame, img_shape), dtype=np.float32)
    video_list_1 = np.empty((len(videos), true_ceil(max_frame/2), img_shape), dtype=np.float32)
    video_list_2 = np.empty((len(videos), true_floor(max_frame/2), img_shape), dtype=np.float32)

    j = 0

    for video in videos:

        frames = os.listdir(f"{source}/{label}/{video}")
        k = 0

        if "desktop.ini" in frames:
            frames.remove("desktop.ini")

        count_frame = 0

        for frame in frames:

            num_frames = len(os.listdir(f"{source}/{label}/{video}"))

            #USE OPENCV
            img = load_img(f"{source}/{label}/{video}/{frame}", color_mode="grayscale")
            img = np.asarray(img, dtype=np.float32)
            img /= 255.
            img = img.flatten()

            video_list[j, k, :] = img
            if count_frame < true_ceil(num_frames/2):
                video_list_1[j, k, :] = img
                count_frame += 1
            else:

                video_list_2[j, k-count_frame, :] = img

            k += 1

        # Zero-padding
        while k < max_frame: # video_list.shape[1] < max_frame:
            video_list[j, k, :] = zeropad

            if video_list_1.shape[1] < true_ceil(max_frame/2):
                video_list_1[j,k,:] = zeropad
                count_frame += 1
            elif k-count_frame < true_floor(max_frame/2):
                video_list_2[j,k-count_frame,:] = zeropad

            k += 1

        j += 1

    label_list = np.array([num_class[0] for i in range(len(video_list))])
    num_class[0] += 1

    print(f"Label {label} has been imported. There were {len(video_list)} videos in it.")
    # print(f"It took {time()-start[1]}s")
    start[1] = time()

    return video_list, label_list, video_list_1, video_list_2


nb_vid = 0

for label in os.listdir(f"{source}"):
    nb_vid += len(os.listdir(f"{source}/{label}"))

labels = os.listdir(f"{source}")

list_arrays_x = []
list_arrays_x1 = []
list_arrays_x2 = []
list_labels = []

for label in labels:
    video_x, video_y, video_x1, video_x2 = import_data1(source, label, num_classes)
    list_arrays_x.append(video_x)
    del video_x
    list_arrays_x1.append(video_x1)
    del video_x1
    list_arrays_x2.append(video_x2)
    del video_x2
    list_labels.append(video_y)
    del video_y


x_train0 = np.concatenate(list_arrays_x, axis=0)
del list_arrays_x
x_train1 = np.concatenate(list_arrays_x1, axis=0)
del list_arrays_x1
x_train2 = np.concatenate(list_arrays_x2, axis=0)
del list_arrays_x2

#x_train_conc012 = np.concatenate([x_train0, x_train1, x_train2], axis=1)
#del x_train0, x_train1, x_train2

y_train = np.concatenate(list_labels, axis=0)
del list_labels


x_train0.swapaxes(1, 2)
x_train1.swapaxes(1, 2)
x_train2.swapaxes(1, 2)
#x_train = np.expand_dims(x_train, -1)
y_train = np.expand_dims(y_train, -1)

print(f"Finished importing data. It took {time() - start[0]}s")


#print(x_train.shape)
print(y_train.shape)


# opt_adam = tf.keras.optimizers.Adam(learning_rate=0.00025)
save_best = ModelCheckpoint(filepath="best_tri", monitor='loss', save_best_only=True,
                            save_freq='epoch', save_weights_only=True, verbose=1)
save_val_best = ModelCheckpoint(filepath="best_val_tri", monitor='val_loss', save_best_only=True,
                                save_freq='epoch', save_weights_only=True, verbose=1)

stopping = EarlyStopping(monitor="val_loss", patience=30)

#print(x_train.shape[1:])
input_sequence = Input(x_train0.shape[1:])
input_sequence_half1 = Input(x_train1.shape[1:])
input_sequence_half2 = Input(x_train2.shape[1:])

cell = SRUCell(num_stats=40, mavg_alphas=[0.0, 0.5, 0.9, 0.99, 0.999], recur_dims=10)
cell1 = SRUCell(num_stats=25, mavg_alphas=[0.0, 0.5, 0.99, 0.999], recur_dims=10)
cell2 = SRUCell(num_stats=25, mavg_alphas=[0.0, 0.5, 0.99, 0.999], recur_dims=10)

rnn = RNN([cell], return_sequences=False)(input_sequence)
rnn1 = RNN([cell1], return_sequences=False)(input_sequence_half1)
rnn2 = RNN([cell2], return_sequences=False)(input_sequence_half2)

intermediaire = Dense(32, activation='relu')(keras.layers.concatenate([rnn, rnn1, rnn2]))

output = Dense(num_classes[0], activation='softmax')(intermediaire)

model = Model(inputs=[input_sequence, input_sequence_half1, input_sequence_half2], outputs=[output])

model.load_weights("best_tri")
opt = SGD(learning_rate=0.05)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#model.summary()

def unison_shuffled_copies(a, b, c, d):
    assert len(a) == len(b) == len(c) == len(d)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]


x_train0, x_train1, x_train2, y_train = unison_shuffled_copies(x_train0, x_train1, x_train2, y_train)



with device("/cpu:0"):
    model.evaluate([x_train0, x_train1, x_train2], y_train, batch_size=16, verbose=1)

    # history = model.fit([x_train0, x_train1, x_train2], y_train, batch_size=32, epochs=500, verbose=1,
    #                     validation_split=0.1, callbacks=[save_best, save_val_best, stopping])

