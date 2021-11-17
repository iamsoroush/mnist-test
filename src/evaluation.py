import tensorflow.keras as tfk

from abstractions import EvaluatorBase


class Evaluator(EvaluatorBase):
    def get_eval_funcs(self):
        return {'sparse categorical ce': self.get_cat_loss()}

    def get_cat_loss(self, from_logits=False):

        def scce_loss(y_true, y_pred):
            scce = tfk.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
            return scce(y_true, y_pred).numpy()

        return scce_loss
