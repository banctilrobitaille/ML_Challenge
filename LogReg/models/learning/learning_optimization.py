import numpy as np


class OptimizationType(object):
    GRADIENT = "gradient"


class GradientDescent(object):
    @staticmethod
    def compute_gradient(probability_matrix, target_matrix, feature_matrix):
        return -(np.dot(feature_matrix.T, (target_matrix - probability_matrix))) / feature_matrix.shape[0]


class UpdateWeights(object):
    @staticmethod
    def update_weights(weight_matrix, probability_matrix, target_matrix, feature_matrix, learning_rate):
        weight_matrix -= learning_rate * GradientDescent.compute_gradient(probability_matrix, target_matrix,
                                                                          feature_matrix)
