import os
import glob
import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa
from pprint import pprint


def get_classes():
    classes = ["Cassava Bacterial Blight",
               "Cassava Brown Streak Disease",
               "Cassava Green Mottle",
               "Cassava Mosaic Disease",
               "Healthy"]
    return classes


def _standardization_(crop, label):
    # crop = tf.image.per_image_standardization(crop)
    crop = crop / 255
    return crop, label


def _random_jitter_(crop, l):
    crop = tf.image.resize(crop, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # crop = tf.image.random_crop(crop, size=[150, 150, 3])
    crop = tf.image.random_flip_up_down(crop)
    crop = tf.image.random_flip_left_right(crop)
    crop = tf.image.rot90(crop)


    return crop, l

def _augmentations_(crop):
    crop = crop.numpy()
    seq = iaa.Sequential([(
        iaa.Affine(scale=(1.0, 1.1),
                   translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                   # rotate=(-45, 45),
                   shear=(-1, 1),
                   mode='constant'))])
    crop = seq.augment_image(crop)
    return crop


def _training_augmentation_(crop, label):
    shape = crop.get_shape()
    crop = tf.py_function(_augmentations_, inp=[crop], Tout=tf.uint8)
    crop.set_shape(shape)
    crop = _random_jitter_(crop)
    return crop, label


class DataLoader(object):
    def __init__(self, data_path, training=True):
        self.data_path = data_path
        self.training = 'train' if training else 'validation'
        self.classes = get_classes()
        self.seed = 1001
        if self.training == 'train':
            self.batch_size =64
            self.buffer = 1000
        else:
            self.batch_size = 64
            self.buffer = 100

    def parse_record(self, record):
        features = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'height': tf.io.FixedLenFeature([], dtype=tf.int64),
            'width': tf.io.FixedLenFeature([], dtype=tf.int64),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64),
        }
        record = tf.io.parse_single_example(record, features)
        img = tf.io.decode_raw(record['image'], tf.float32)
        img = tf.reshape(img, [record['height'], record['width'], 3])
        # label = record['label']
        label = tf.one_hot(record['label'], len(self.classes), dtype=tf.float32)
        return img, label

    def train_dataset(self, label):

        files = os.path.join(self.data_path, '{}_{}*.tf_record'.format(self.training, label))
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=True, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn), cycle_length=len(filenames), num_parallel_calls=min(len(filenames), tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.training == 'train':
            dataset = dataset.map(_random_jitter_, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(_standardization_, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.shuffle(self.buffer, seed=self.seed)
            dataset = dataset.repeat()
        return dataset

    def balanced_train_dataset(self):
        datasets = []
        for cl in self.classes:
            datasets.append(self.train_dataset(cl))
        importance = [0.2, 0.2, 0.2, 0.2, 0.2]
        sampled_dataset = tf.data.experimental.sample_from_datasets(datasets, weights=importance)
        sampled_dataset = sampled_dataset.batch(self.batch_size)
        sampled_dataset = sampled_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return sampled_dataset

    def validation_dataset(self):
        files = os.path.join(self.data_path, '{}_*.tf_record'.format(self.training))
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=True, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn), cycle_length=len(filenames), num_parallel_calls=min(len(filenames), tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(_standardization_, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(self.batch_size)
        return dataset



if __name__ == '__main__':
    train_dataset = DataLoader("D:\\PROJECTS\Casava_Classification\\DataSet\\tf_records_224", training=True)
    validation_dataset = DataLoader("D:\\PROJECTS\Casava_Classification\\DataSet\\tf_records_224", training=False)
    print(train_dataset)


    for i, batch in enumerate(train_dataset.balanced_train_dataset()):
        images, labels = batch
        if i > 0:
            break

        print('images size: {}, labels size: {}'.format(images.shape, labels.shape))


        labels = labels.numpy()
        labels = np.argmax(labels, axis=1)
        elements, counts = np.unique(labels, return_counts=True)
        print(elements, counts)
        pprint('elements: {}, counts: {}'.format(elements, counts))

