import tensorflow.keras as tfk
import tensorflow as tf

from abstractions import ModelBuilderBase


class ModelBuilder(ModelBuilderBase):
    def get_compiled_model(self):
        model = tfk.models.Sequential([
            tfk.layers.Dense(self.n_filters, activation=self.activation, input_shape=(self.input_h * self.input_w,)),
            tfk.layers.Dropout(self.dropout_rate),
            tfk.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.metrics.SparseCategoricalAccuracy()])

        return model

    def _load_params(self, config):
        m_config = config.model_builder
        self.activation = m_config.activation
        self.n_filters = m_config.n_filters
        self.dropout_rate = m_config.dropout_rate
        self.input_h = config.input_height
        self.input_w = config.input_width

    def _set_defaults(self):
        self.activation = 'relu'
        self.n_filters = 512
        self.dropout_rate = 0.2
        self.input_h = 28
        self.input_w = 28
