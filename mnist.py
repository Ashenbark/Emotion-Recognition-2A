'''Trains a SRU on the MNIST dataset.

The parameters are not optimized.
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Model
from keras.layers import Dense, RNN, Input
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint


from SRU import SRUCell

num_classes = 10

# input image dimensions
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train=x_train[0:2000]
y_train=y_train[0:2000]
x_test=x_test[0:1000]
y_test=y_test[0:1000]

opt_adam = tf.keras.optimizers.Adam(learning_rate=0.00025)
save_best = ModelCheckpoint(filepath="best", monitor='loss', save_best_only=True,
                            save_freq='epoch', save_weights_only=True, verbose=1)
save_val_best = ModelCheckpoint(filepath="best_val", monitor='val_loss', save_best_only=True,
                            save_freq='epoch', save_weights_only=True, verbose=1)

inputs = Input(x_train.shape[1:])
cell = SRUCell(num_stats=50, mavg_alphas=[0.0,0.5,0.9,0.99], recur_dims=20)
rnn = RNN([cell], return_sequences=False)(inputs)
output = Dense(num_classes, activation='softmax')(rnn)

model = Model(inputs=[inputs], outputs=[output])
model.load_weights("best")
model.compile(loss='categorical_crossentropy',optimizer=opt_adam,metrics=['accuracy'])


#model.fit(x_train, y_train, batch_size=24, epochs=50, verbose=1,
#          validation_data=(x_test, y_test), callbacks=[save_best, save_val_best])
"""
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
"""

img = keras.preprocessing.image.load_img("4.png", color_mode="grayscale")
img = np.asarray(img)
plt.figure()
plt.imshow(img)
plt.show()

img = img.reshape(1, -1)
img= img/255
img=1-img
print(img.shape)
n = 750
xxx = x_test[n,:,:].reshape(1, -1)
print(xxx.shape)
print(y_test[n])

pred = model.predict(img)

print(np.argmax(pred))