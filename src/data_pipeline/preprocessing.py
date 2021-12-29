from abstractions.utils import ConfigStruct
import tensorflow as tf
import numpy as np

from abstractions import PreprocessorBase


class PreprocessorTF(PreprocessorBase):
    def add_image_preprocess(self, generator):
        return generator.map(self._wrapper_image_preprocess)

    def add_label_preprocess(self, generator):
        return generator

    def add_weight_preprocess(self, generator):
        return generator

    def batchify(self, generator, n_data_points):
        gen = generator.batch(self.batch_size).repeat()
        n_iter = n_data_points // self.batch_size + int((n_data_points % self.batch_size) > 0)
        return gen, n_iter

    def image_preprocess(self, image):
        return tf.reshape(image, (self.input_h * self.input_w,)) / self.normalize_by

    def _wrapper_image_preprocess(self, x, y, w):
        pre_processed = self.image_preprocess(x)
        return pre_processed, y, w

    def _load_params(self, config: ConfigStruct):
        self.normalize_by = config.preprocessor.normalize_by
        self.input_h = config.input_height
        self.input_w = config.input_width
        self.batch_size = config.batch_size

    def _set_defaults(self):
        self.normalize_by = 255
        self.input_h = 28
        self.input_w = 28
        self.batch_size = 8


class Preprocessor(PreprocessorBase):
    def add_image_preprocess(self, generator):
        while True:
            x, y, w = next(generator)
            yield self.image_preprocess(x), y, w

    def add_label_preprocess(self, generator):
        return generator

    def add_weight_preprocess(self, generator):
        return generator

    def batchify(self, generator, n_data_points):
        n_iter = n_data_points // self.batch_size + int((n_data_points % self.batch_size) > 0)
        gen = self._batch_gen(generator, self.batch_size)
        return gen, n_iter

    def image_preprocess(self, image):
        return np.reshape(image, (self.input_h * self.input_w)) / self.normalize_by

    @staticmethod
    def _batch_gen(generator, batch_size):
        while True:
            x_b, y_b, z_b = list(), list(), list()
            for i in range(batch_size):
                x, y, z = next(generator)
                x_b.append(x)
                y_b.append(y)
                z_b.append(z)
            yield np.array(x_b), np.array(y_b), np.array(z_b)

    def _load_params(self, config: ConfigStruct):
        self.normalize_by = config.preprocessor.normalize_by
        self.input_h = config.input_height
        self.input_w = config.input_width
        self.batch_size = config.batch_size

    def _set_defaults(self):
        self.normalize_by = 255
        self.input_h = 28
        self.input_w = 28
        self.batch_size = 8
