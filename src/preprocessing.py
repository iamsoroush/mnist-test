import tensorflow as tf

from abstractions import PreprocessorBase


class PreprocessorTF(PreprocessorBase):
    def image_preprocess(self, image):
        return tf.reshape(image, (self.input_h * self.input_w,)) / self.normalize_by

    def _wrapper_image_preprocess(self, x, y, w):
        pre_processed = self.image_preprocess(x)
        return pre_processed, y, w

    def label_preprocess(self, label):
        return label

    def add_image_preprocess(self, generator):
        return generator.map(self._wrapper_image_preprocess)

    def add_label_preprocess(self, generator):
        return generator

    def batchify(self, generator, n_data_points, batch_size):
        gen = generator.batch(batch_size).repeat()
        n_iter = n_data_points // batch_size + int((n_data_points % batch_size) > 0)
        return gen, n_iter

    def _load_params(self, config):
        self.normalize_by = config.preprocessor.normalize_by
        self.input_h = config.input_height
        self.input_w = config.input_width

    def _set_defaults(self):
        self.normalize_by = 255
        self.input_h = 28
        self.input_w = 28
