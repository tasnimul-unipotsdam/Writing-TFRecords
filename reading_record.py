import os
import glob
import tensorflow as tf


def get_classes():
    classes = ["AnnualCrop",
               "Forest",
               "HerbaceousVegetation",
               "Highway",
               "Industrial",
               "Pasture",
               "PermanentCrop",
               "Residential",
               "River",
               "SeaLake"]
    return classes


class DataLoader(object):
    def __init__(self, data_path, training=True):
        self.data_path = data_path
        self.training = 'train' if training else 'validation'
        self.classes = get_classes()
        self.seed = 1
        if self.training == 'train':
            self.batch_size = 128
            self.buffer = 1000
        else:
            self.batch_size = 128
            self.buffer = 100

    def parse_record(self, record):
        features = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'height': tf.io.FixedLenFeature([], dtype=tf.int64),
            'width': tf.io.FixedLenFeature([], dtype=tf.int64),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64),
        }
        record = tf.io.parse_single_example(record, features)
        img = tf.io.decode_raw(record['image'], tf.uint8)
        img = tf.reshape(img, [record['height'], record['width'], 3])
        label = tf.one_hot(record['label'], len(self.classes), dtype=tf.float32)
        return img, label

    def agumentation(self, crop, label):
        crop = tf.image.random_flip_up_down(crop)
        crop = tf.image.random_flip_left_right(crop)
        crop = tf.image.rot90(crop)
        crop = tf.image.random_brightness(crop, max_delta=.25)
        crop = tf.cast(crop, dtype=tf.float32)
        noise = tf.random.normal(shape=tf.shape(crop), mean=0, stddev=1, dtype=tf.float32)
        crop = tf.add(noise, crop)
        crop = crop / 255
        return crop, label

    def val_aug(self, crop, label):
        crop = crop/255
        return crop, label

    def load_dataset(self, label):
        files = os.path.join(self.data_path, '{}_{}*.tfrecord'.format(self.training, label))
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=True, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn), cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames), tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.training == 'train':
            dataset = dataset.map(self.agumentation)
        else:
            dataset = dataset.map(self.val_aug)

        dataset = dataset.shuffle(self.buffer, seed=self.seed)
        dataset = dataset.repeat()
        return dataset

    def balanced_batch(self):
        datasets = []
        for cl in self.classes:
            datasets.append(self.load_dataset(cl))
        importance = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        sampled_dataset = tf.data.experimental.sample_from_datasets(datasets, weights=importance)
        sampled_dataset = sampled_dataset.batch(self.batch_size)
        sampled_dataset = sampled_dataset.prefetch(2)
        return sampled_dataset

    def general_dataset(self):

        files = os.path.join(self.data_path, '{}_*.tfrecord'.format(self.training))
        filenames = glob.glob(files)
        dataset = tf.data.Dataset.list_files(files, shuffle=True, seed=self.seed)
        dataset = dataset.interleave(lambda fn: tf.data.TFRecordDataset(fn), cycle_length=len(filenames),
                                     num_parallel_calls=min(len(filenames), tf.data.experimental.AUTOTUNE))
        dataset = dataset.map(self.parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.training == 'train':
            dataset = dataset.map(self.agumentation)
        else:
            dataset = dataset.map(self.val_aug)
        dataset = dataset.shuffle(self.buffer, seed=self.seed)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(self.batch_size)
        return dataset


if __name__ == '__main__':
    train_dataset = DataLoader("D:\\PROJECTS\\Writing-TFRecords\\records", training=True).general_dataset()
    validation_dataset = DataLoader("D:\\PROJECTS\\Writing-TFRecords\\records", training=False).general_dataset()

