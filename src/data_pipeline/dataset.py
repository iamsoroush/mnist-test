import pathlib

import numpy as np
import tensorflow as tf
import skimage.io
import pandas as pd

from abstractions import DataLoaderBase


class DataLoaderTF(DataLoaderBase):

    def __init__(self, config, data_dir):
        super().__init__(config, data_dir)

        self.train_data_dir = self.data_dir.joinpath('train')
        self.val_data_dir = self.data_dir.joinpath('validation')
        self.test_data_dir = self.data_dir.joinpath('test')

    def create_training_generator(self):
        labels_df = self._read_labels_df(self.train_data_dir)
        n_data = len(labels_df)

        img_paths = tf.data.Dataset.from_tensor_slices(labels_df['path'])
        images = img_paths.map(self._read_image_tf)
        labels = tf.data.Dataset.from_tensor_slices(labels_df['label'])
        weights = tf.data.Dataset.from_tensor_slices(tf.ones(n_data))

        train_ds = tf.data.Dataset.zip((images, labels, weights))
        if self.shuffle:
            train_ds = train_ds.shuffle(n_data)

        return train_ds, n_data

    def create_validation_generator(self):
        labels_df = self._read_labels_df(self.val_data_dir)
        n_data = len(labels_df)

        img_paths = tf.data.Dataset.from_tensor_slices(labels_df['path'])
        images = img_paths.map(self._read_image_tf)
        labels = tf.data.Dataset.from_tensor_slices(labels_df['label'])
        weights = tf.data.Dataset.from_tensor_slices(tf.ones(n_data))

        ds = tf.data.Dataset.zip((images, labels, weights))

        return ds, n_data

    def create_test_generator(self):
        labels_df = self._read_labels_df(self.test_data_dir)
        n_data = len(labels_df)

        img_paths = tf.data.Dataset.from_tensor_slices(labels_df['path'])
        images = img_paths.map(self._read_image_tf)
        labels = tf.data.Dataset.from_tensor_slices(labels_df['label'])
        data_ids = tf.data.Dataset.from_tensor_slices(labels_df['image name'])

        ds = tf.data.Dataset.zip((images, labels, data_ids))

        return ds, n_data

    def get_validation_index(self):
        return self._read_labels_df(self.val_data_dir)['image name'].values

    @staticmethod
    def _read_labels_df(data_dir: pathlib.Path):
        labels_df = pd.read_csv(data_dir.joinpath('labels.csv'))
        labels_df['path'] = labels_df.apply(lambda x: str(data_dir.joinpath(x['image name'])), axis=1)
        return labels_df

    def _load_params(self, config):
        self.shuffle = config.data_loader.shuffle

    def _set_defaults(self):
        self.shuffle = True

    @staticmethod
    def _read_image_tf(path):
        return tf.image.decode_jpeg(tf.io.read_file(path))


class DataLoader(DataLoaderBase):

    def __init__(self, config, data_dir):
        super().__init__(config, data_dir)

        self.train_data_dir = self.data_dir.joinpath('train')
        self.val_data_dir = self.data_dir.joinpath('validation')
        self.test_data_dir = self.data_dir.joinpath('test')

    def create_training_generator(self):
        labels_df = self._read_labels_df(self.train_data_dir)
        n_data = len(labels_df)

        gen = self._generator(labels_df['path'], labels_df['label'], self.shuffle)
        return gen, n_data

    def create_validation_generator(self):
        labels_df = self._read_labels_df(self.val_data_dir)
        n_data = len(labels_df)

        gen = self._generator(labels_df['path'], labels_df['label'], False)
        return gen, n_data

    def create_test_generator(self):
        labels_df = self._read_labels_df(self.test_data_dir)
        n_data = len(labels_df)

        gen = self._test_gen(labels_df['path'], labels_df['label'], labels_df['image name'])
        return gen, n_data

    def get_validation_index(self):
        return self._read_labels_df(self.val_data_dir)['image name'].values

    @staticmethod
    def _read_labels_df(data_dir: pathlib.Path):
        labels_df = pd.read_csv(data_dir.joinpath('labels.csv'))
        labels_df['path'] = labels_df.apply(lambda x: str(data_dir.joinpath(x['image name'])), axis=1)
        return labels_df

    def _load_params(self, config):
        self.shuffle = config.data_loader.shuffle

    def _set_defaults(self):
        self.shuffle = True

    @staticmethod
    def _read_image(path):
        """returns (28, 28)"""
        return skimage.io.imread(path, as_gray=True)

    def _generator(self, image_paths, label_array, shuffle):
        indxs = np.arange(len(image_paths))
        while True:
            if shuffle:
                np.random.shuffle(indxs)
            for ind in indxs:
                image = self._read_image(image_paths[ind])
                yield image, label_array[ind], 1

    def _test_gen(self, image_paths, label_array, data_ids):
        for img_path, label, data_id in zip(image_paths, label_array, data_ids):
            image = self._read_image(img_path)
            yield image, label, data_id
