import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_label(label):
    if label == "Cassava Bacterial Blight":
        return 0
    if label == "Cassava Brown Streak Disease":
        return 1
    if label == 'Cassava Green Mottle':
        return 2
    if label == "Cassava Mosaic Disease":
        return 3
    if label == "Healthy":
        return 4

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


def shard_dataset(dataset, num_records=20):
    chunk = len(dataset) // num_records
    parts = [(k * chunk) for k in range(len(dataset)) if (k * chunk) < len(dataset)]
    return chunk, parts


class WritingTFRecord(object):
    def __init__(self, data_path, **kwargs):
        self.data_path = data_path
        self.other_arguments = kwargs

    def load_dataset(self, data_type='train'):
        img_dir = os.path.join(self.data_path, data_type)
        labels = os.listdir(img_dir)
        data = []
        for label in tqdm(labels):
            list_image_fns = glob.glob(os.path.join(img_dir, label, '*'))
            for fn in list_image_fns:
                image = cv2.imread(fn, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                lab_planes = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab_planes[0] = clahe.apply(lab_planes[0])
                merge = cv2.merge(lab_planes)
                img = cv2.cvtColor(merge, cv2.COLOR_LAB2RGB)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                img = img.astype(np.float32)
                meta = {
                    'image': img,
                    'label': label,
                    'data_type': data_type
                }
                data.append(meta)

        print('Finished reading for {} from the array....'.format(data_type))
        return data

    def save_data(self, dataset, label, data_name='Cassava', data_type='train'):
        train_check = 0
        if len(dataset) > 100:
            chunk, parts = shard_dataset(dataset)
            for i, j in enumerate(tqdm(parts)):
                shard = dataset[j:(j + chunk)]
                fn = '{}_{}-{}_{:03d}-{:03d}.tf_record'.format(data_type, label, data_name, i + 1, len(parts))
                _process_examples(shard, os.path.join(self.data_path, 'tf_records_224', fn))
                train_check += len(shard)
            print('Number of saved samples for {}: {}'.format(label, train_check))
        else:
            fn = '{}_{}-{}_{:03d}-{:03d}.tf_record'.format(data_type, label, data_name, 1, 1)
            _process_examples(dataset, os.path.join(self.data_path, 'tf_records_224', fn))
            print('Small dataset with {} samples'.format(len(dataset)))
        return None

    def preprocess_data(self):
        for d in ['validation', 'train']:
            print("Working for {}...".format(d))
            data = self.load_dataset(data_type=d)
            labels = list(set([ex['label'] for ex in data]))
            for l in labels:
                label_data = [ex for ex in data if ex['label'] == l]
                self.save_data(label_data, l, data_type=d)


if __name__ == '__main__':
    prep = WritingTFRecord("D:\\PROJECTS\\Casava_Classification\\DataSet")
    prep.preprocess_data()
