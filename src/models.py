import tensorflow.keras as tfk
import tensorflow as tf

from abstractions import ModelBuilderBase


class ModelBuilder(ModelBuilderBase):
    def get_compiled_model(self):
        model = tfk.models.Sequential([
            tfk.layers.Dense(self.n_filters, activation=self.activation, input_shape=(784,)),
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

    def _set_defaults(self):
        self.activation = 'relu'
        self.n_filters = 512
        self.dropout_rate = 0.2
