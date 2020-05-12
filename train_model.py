import os
import random

import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from reading_record import DataLoader

random.seed(1001)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

with tf.device("/gpu:0"):
    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = inputs
    x = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_regularizer=l2(.0001))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_regularizer=l2(.0001))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", kernel_regularizer=l2(.0001))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", kernel_regularizer=l2(.0001))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", kernel_regularizer=l2(.0001))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", kernel_regularizer=l2(.0001))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(512, 3, padding="same", kernel_regularizer=l2(.0001))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(512, 3, padding="same", kernel_regularizer=l2(.0001))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(64, kernel_regularizer=l2(.0001))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, x, name='model')

model.summary()
log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=["accuracy"])

train_dataset = DataLoader("D:\\PROJECTS\\EuroSAT\\records", training=True).general_dataset()
validation_dataset = DataLoader("D:\\PROJECTS\\EuroSAT\\records", training=False).general_dataset()

dict_file = open("weights.json", "r")
class_weight = dict_file.read()

history = model.fit(train_dataset, verbose=2, epochs=150, class_weight=class_weight,
                    validation_data=validation_dataset, callbacks=[board])

train_score = model.evaluate(train_dataset, verbose=0)
print('train loss, train acc:', train_score)

validation_score = model.evaluate(validation_dataset, verbose=0)
print('validation loss, validation acc:', validation_score)

model.save("model.h5")

fig = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
fig.savefig("accuracy_and_loss")
