import random
#from tensorflow.keras.preprocessing.image import load_img
import cv2
from time import time
import os
import numpy as np

max_frame = 141
img_shape = 128 * 128
num_classes = 7

start = [time(), time()]
print("Starting importing data...")


def true_ceil(x):
    return int(np.ceil(x))


def true_floor(x):
    return int(np.floor(x))


def importDataFromLabel(source, label, current_class):
    zeropad = np.zeros(img_shape)

    videos = os.listdir(f"{source}/{label}")
    random.shuffle(videos)
    video_list = np.empty((len(videos), max_frame, img_shape), dtype=np.float32)
    video_list_1 = np.empty((len(videos), true_ceil(max_frame / 2), img_shape), dtype=np.float32)
    video_list_2 = np.empty((len(videos), true_floor(max_frame / 2), img_shape), dtype=np.float32)

    j = 0

    for video in videos:

        frames = os.listdir(f"{source}/{label}/{video}")
        k = 0

        if "desktop.ini" in frames:
            frames.remove("desktop.ini")

        count_frame = 0

        for frame in frames:

            num_frames = len(os.listdir(f"{source}/{label}/{video}"))

            # USE OPENCV
            img = cv2.imread(f"{source}/{label}/{video}/{frame}", cv2.IMREAD_GRAYSCALE)
            img = np.asarray(img, dtype=np.float32)
            img /= 255.
            img = img.flatten()

            video_list[j, k, :] = img
            if count_frame < true_ceil(num_frames / 2):
                video_list_1[j, k, :] = img
                count_frame += 1
            else:

                video_list_2[j, k - count_frame, :] = img

            k += 1

        # Zero-padding
        while k < max_frame:  # video_list.shape[1] < max_frame:
            video_list[j, k, :] = zeropad

            if video_list_1.shape[1] < true_ceil(max_frame / 2):
                video_list_1[j, k, :] = zeropad
                count_frame += 1
            elif k - count_frame < true_floor(max_frame / 2):
                video_list_2[j, k - count_frame, :] = zeropad

            k += 1

        j += 1

    label_list = np.array([current_class for i in range(len(video_list))])

    print(f"Label {label} has been imported. There were {len(video_list)} videos in it.")
    # print(f"It took {time()-start[1]}s")
    start[1] = time()

    return video_list, label_list, video_list_1, video_list_2


def importDataFromSource(source):
    nb_vid = 0

    for label in os.listdir(f"{source}"):
        nb_vid += len(os.listdir(f"{source}/{label}"))

    labels = os.listdir(f"{source}")

    list_arrays_x = []
    list_arrays_x1 = []
    list_arrays_x2 = []
    list_labels = []
    current_class = 0
    # Labels are the name, current_class contains the ID (int) of the current label

    for label in labels:
        # Using the previous function, we important for each label/class the corresponding videos
        video_x, video_y, video_x1, video_x2 = importDataFromLabel(source, label, current_class)
        current_class += 1

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


    y_train = np.concatenate(list_labels, axis=0)
    del list_labels

    x_train0.swapaxes(1, 2)
    x_train1.swapaxes(1, 2)
    x_train2.swapaxes(1, 2)
    # x_train = np.expand_dims(x_train, -1)
    y_train = np.expand_dims(y_train, -1)

    print(f"Finished importing data. It took {time() - start[0]}s")

    print(x_train0.shape, x_train1.shape, x_train2.shape)
    # print(y_train.shape)

    return x_train0, x_train1, x_train2, y_train


def importDataFromLabelSingle(source, label, current_class):
    zeropad = np.zeros(img_shape)

    videos = os.listdir(f"{source}/{label}")
    random.shuffle(videos)
    video_list = np.empty((len(videos), max_frame, img_shape), dtype=np.float32)

    j = 0

    for video in videos:

        frames = os.listdir(f"{source}/{label}/{video}")
        k = 0

        if "desktop.ini" in frames:
            frames.remove("desktop.ini")

        for frame in frames:

            num_frames = len(os.listdir(f"{source}/{label}/{video}"))

            # USE OPENCV
            img = cv2.imread(f"{source}/{label}/{video}/{frame}", cv2.IMREAD_GRAYSCALE)
            img = np.asarray(img, dtype=np.float32)
            img /= 255.
            img = img.flatten()

            video_list[j, k, :] = img
            k += 1

        # Zero-padding
        while k < max_frame:  # video_list.shape[1] < max_frame:
            video_list[j, k, :] = zeropad
            k += 1

        j += 1

    label_list = np.array([current_class for i in range(len(video_list))])

    print(f"Label {label} has been imported. There were {len(video_list)} videos in it.")
    # print(f"It took {time()-start[1]}s")
    start[1] = time()

    return video_list, label_list


def importDataFromSourceSingle(source):
    nb_vid = 0

    for label in os.listdir(f"{source}"):
        nb_vid += len(os.listdir(f"{source}/{label}"))

    labels = os.listdir(f"{source}")

    list_data = []
    list_labels = []
    current_class = 0
    # Labels are the name, current_class contains the ID (int) of the current label

    for label in labels:
        # Using the previous function, we important for each label/class the corresponding videos
        video_x, video_y = importDataFromLabelSingle(source, label, current_class)
        current_class += 1

        list_data.append(video_x)
        del video_x
        list_labels.append(video_y)
        del video_y

    x_train = np.concatenate(list_data, axis=0)
    del list_data
    y_train = np.concatenate(list_labels, axis=0)
    del list_labels

    x_train = x_train.swapaxes(1, 2)
    y_train = np.expand_dims(y_train, -1)

    print(f"Finished importing data. It took {time() - start[0]}s")

    return x_train, y_train


def unison_shuffled_copies(a, b, c, d):
    np.random.seed(2023)
    assert len(a) == len(b) == len(c) == len(d)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]
