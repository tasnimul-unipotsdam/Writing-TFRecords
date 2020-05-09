import os
import glob
import json
import cv2
from pprint import pprint

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tqdm import tqdm

from collections import Counter


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_label(label):
    if label == "AnnualCrop":
        return 0
    if label == "Forest":
        return 1
    if label == 'HerbaceousVegetation':
        return 2
    if label == "Highway":
        return 3
    if label == "Industrial":
        return 4
    if label == "Pasture":
        return 5
    if label == "PermanentCrop":
        return 6
    if label == "Residential":
        return 7
    if label == "River":
        return 8
    if label == "SeaLake":
        return 9


def _process_examples(example_data, filename: str, channels=3):

    with tf.io.TFRecordWriter(filename) as writer:
        for i, ex in enumerate(example_data):
            image = ex['image'].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(ex['image'].shape[0]),
                'width': _int64_feature(ex['image'].shape[1]),
                'depth': _int64_feature(channels),
                'image': _bytes_feature(image),
                'label': _int64_feature(encode_label(ex['label']))
            }))
            writer.write(example.SerializeToString())
    return None


class WritingTFRecord(object):
    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        self.other_arguments = kwargs

    def load_dataset(self, data_type='train'):
        img_dir = os.path.join(self.data_path, data_type)
        labels = os.listdir(img_dir)
        data = []
        for l in tqdm(labels):
            list_image_fns = glob.glob(os.path.join(img_dir, l, '*'))
            for fn in list_image_fns:
                img = cv2.imread(fn, cv2.IMREAD_COLOR)
                meta = {
                    'image': img,
                    'label': l,
                    'data_type': data_type
                }
                data.append(meta)
        """Assigning Class Weights for Imbalanced Dataset"""
        y = np.array([ex['label'] for ex in data])
        class_weights = class_weight.compute_class_weight("balanced", np.unique(y), y)
        class_weights = dict(enumerate(class_weights))
        print(class_weights)
        dict_file = open("weights.json", "w")
        json.dump(class_weights, dict_file)
        dict_file.close()
        pprint('number of samples in {}: {}'.format(data_type, len(data)))
        pprint('data statistics {}'.format(dict(Counter([ex['label'] for ex in data]))))
        return data

    def shard_dataset(self, dataset, num_records=20):
        chunk = len(dataset) // num_records
        parts = [(k * chunk) for k in range(len(dataset)) if (k * chunk) < len(dataset)]
        return chunk, parts

    def save_data(self, dataset, label, dataname='euro_sat', data_type='train'):
        train_check = 0
        if len(dataset) > 100:
            chunk, parts = self.shard_dataset(dataset)
            for i, j in enumerate(tqdm(parts)):
                shard = dataset[j:(j + chunk)]
                fn = '{}_{}-{}_{:03d}-{:03d}.tfrecord'.format(data_type, label, dataname, i + 1, len(parts))
                _process_examples(shard, os.path.join(self.data_path, 'records', fn))
                train_check += len(shard)
            print('Number of saved samples for {}: {}'.format(label, train_check))
        else:
            fn = '{}_{}-{}_{:03d}-{:03d}.tfrecord'.format(data_type, label, dataname, 1, 1)
            _process_examples(dataset, os.path.join(self.data_path, 'records', fn))
            print('Small dataset with {} samples'.format(len(dataset)))
        return None

    def preprocess_data(self):
        for d in ['train', 'validation', 'test']:
            data = self.load_dataset(data_type=d)
            labels = list(set([ex['label'] for ex in data]))
            for l in labels:
                label_data = [ex for ex in data if ex['label'] == l]
                self.save_data(label_data, l, data_type=d)


if __name__ == '__main__':
    prep = WritingTFRecord("D:\\PROJECTS\\Writing-TFRecords")
    prep.preprocess_data()
