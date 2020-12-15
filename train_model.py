import os
import random

import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from focal_loss import SparseCategoricalFocalLoss
from reading_record import DataLoader
random.seed(1001)

TF_ENABLE_GPU_GARBAGE_COLLECTION = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
devises = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devises[0], True)

train_dataset = DataLoader("D:\\PROJECTS\Casava_Classification\\DataSet\\tf_records_224", training=True).balanced_train_dataset()
validation_dataset = DataLoader("D:\\PROJECTS\Casava_Classification\\DataSet\\tf_records_224", training=False).validation_dataset()

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
])

def compile_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = tf.keras.Model(inputs, x)

    # base_model.trainable = False

    for layer in base_model.layers[:100]:
      layer.trainable =  False
    for layer in base_model.layers[100:]:
      layer.trainable =  True


    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  loss= SparseCategoricalFocalLoss(gamma=2),
                  # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.05),
                  metrics=['accuracy'])
    model.summary()
    return model

def train_model(model):
    log_dir = "D:\\PROJECTS\\Casava_Classification\\logs\\fit\\MobileNetV2_6" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    class_weights = {0: 4.078495660559306, 1: 1.9772791023842917,
                    2: 1.8105308219178082, 3: 0.3226579188281965,
                    4: 1.6736842105263159}

    steps_epoch = np.ceil(21147  / 64)
    valid_steps = np.ceil(100  / 32)

    history = model.fit(train_dataset,
                        epochs=100,
                        verbose=2,
                        callbacks=[board],
                        validation_data=validation_dataset,
                        class_weight=class_weights,
                        steps_per_epoch=steps_epoch,
                        workers= 12
                        )

    model.save("D:\\PROJECTS\\Casava_Classification\\model_plot\\MobileNetV2_6.h5")

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()
    fig.savefig("D:\\PROJECTS\\Casava_Classification\\model_plot\\MobileNetV2_6.jpg")



if __name__ == '__main__':
    compile_model = compile_model()
    train_model(compile_model)
    pass
