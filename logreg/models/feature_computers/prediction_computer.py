import numpy as np
import abc


class ProbabilityType(object):
    SOFTMAX = "softmax"
    PREDICT = "predict"


class ComputeProbability(object):
    __metaclass__ = abc.ABCMeta

    def compute_probability(self, input_matrix):
        raise NotImplementedError

    @classmethod
    def minimize_vector(cls, vector):
        vector -= np.max(vector)
        return vector

    @classmethod
    def probability_vector(cls, vector):
        vector = np.exp(cls.minimize_vector(vector)) / np.sum(np.exp(cls.minimize_vector(vector)))
        return vector

    @classmethod
    def prediction_vector(cls, vector):
        prediction = np.zeros(vector.shape[0])
        prediction[np.argmax(vector)] = 1
        return prediction


class Softmax(ComputeProbability):
    def compute_probability(self, input_matrix):
        softmax_matrix = np.apply_along_axis(ComputeProbability.probability_vector, 1, input_matrix)
        return softmax_matrix


class Predict(ComputeProbability):
    def predict(self, input_matrix):
        softmax_matrix = self.compute_probability(input_matrix)
        prediction_matrix = np.apply_along_axis(ComputeProbability.prediction_vector, 1, softmax_matrix)
        return prediction_matrix

    def compute_probability(self, input_matrix):
        softmax_matrix = np.apply_along_axis(ComputeProbability.probability_vector, 1, input_matrix)
        return softmax_matrix

    @staticmethod
    def prediction_is_true(prediction, target):
        return np.array_equal(prediction, target)


class ProbabilityComputerFactory(object):
    @staticmethod
    def create_probability_computer(probability_type):
        if probability_type == ProbabilityType.SOFTMAX:
            return Softmax()
        elif probability_type == ProbabilityType.PREDICT:
            return Predict()
