import pathlib

import tensorflow as tf
import skimage.io
import pandas as pd

from abstractions import DataLoaderBase


class DataLoaderTF(DataLoaderBase):

    def create_training_generator(self):
        labels_csv = pd.read_csv(self.train_data_dir.joinpath('labels.csv'))
        n_data = len(labels_csv)
        labels_csv['path'] = labels_csv.apply(lambda x: str(self.train_data_dir.joinpath(x['image name'])), axis=1)

        img_paths = tf.data.Dataset.from_tensor_slices(labels_csv['path'])
        images = img_paths.map(self._read_image_tf)
        labels = tf.data.Dataset.from_tensor_slices(labels_csv['label'])
        weights = tf.data.Dataset.from_tensor_slices(tf.ones(n_data))

        train_ds = tf.data.Dataset.zip((images, labels, weights))
        if self.shuffle:
            train_ds = train_ds.shuffle(n_data)

        return train_ds, n_data

    def create_validation_generator(self):
        labels_csv = pd.read_csv(self.val_data_dir.joinpath('labels.csv'))
        n_data = len(labels_csv)
        labels_csv['path'] = labels_csv.apply(lambda x: str(self.val_data_dir.joinpath(x['image name'])), axis=1)

        img_paths = tf.data.Dataset.from_tensor_slices(labels_csv['path'])
        images = img_paths.map(self._read_image_tf)
        labels = tf.data.Dataset.from_tensor_slices(labels_csv['label'])
        weights = tf.data.Dataset.from_tensor_slices(tf.ones(n_data))

        train_ds = tf.data.Dataset.zip((images, labels, weights))

        return train_ds, n_data

    def create_test_generator(self):
        labels_csv = pd.read_csv(self.test_data_dir.joinpath('labels.csv'))
        n_data = len(labels_csv)
        labels_csv['path'] = labels_csv.apply(lambda x: str(self.test_data_dir.joinpath(x['image name'])), axis=1)

        img_paths = tf.data.Dataset.from_tensor_slices(labels_csv['path'])
        images = img_paths.map(self._read_image_tf)
        labels = tf.data.Dataset.from_tensor_slices(labels_csv['label'])
        data_ids = tf.data.Dataset.from_tensor_slices(labels_csv['image name'])

        train_ds = tf.data.Dataset.zip((images, labels, data_ids))

        return train_ds, n_data

    def _load_params(self, config):
        # self.data_dir = pathlib.Path(config.dataset_dir)
        self.train_data_dir = self.data_dir.joinpath('train')
        self.val_data_dir = self.data_dir.joinpath('validation')
        self.test_data_dir = self.data_dir.joinpath('test')
        self.shuffle = config.data_loader.shuffle

    def _set_defaults(self):
        # self.data_dir = pathlib.Path('datasets').joinpath('mnist')
        self.train_data_dir = self.data_dir.joinpath('train')
        self.val_data_dir = self.data_dir.joinpath('validation')
        self.test_data_dir = self.data_dir.joinpath('test')
        self.shuffle = True

    @staticmethod
    def _read_image(path):
        """returns (28, 28)"""
        return skimage.io.imread(path, as_gray=True)

    @staticmethod
    def _read_image_tf(path):
        return tf.image.decode_jpeg(tf.io.read_file(path))
