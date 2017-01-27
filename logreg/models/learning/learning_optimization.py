import numpy as np
from logreg.models.feature_computers.prediction_computer import ProbabilityComputerFactory
from logreg.models.cost_computers.cost_computer import CostComputerFactory


class OptimizationType(object):
    GRADIENT = "gradient"


class GradientDescent(object):
    @classmethod
    def compute_gradient(cls, probability_matrix, target_matrix, feature_matrix):
        return -(np.dot(feature_matrix.T, (target_matrix - probability_matrix))) / feature_matrix.shape[0]


class UpdateWeights(object):
    @staticmethod
    def update_weights(weight_matrix, probability_matrix, target_matrix, feature_matrix, learning_rate):
        weight_matrix -= learning_rate * GradientDescent.compute_gradient(probability_matrix, target_matrix,
                                                                          feature_matrix)
        return weight_matrix


class Learn(object):
    def __init__(self, learning_rate, epoch, cost_threshold, debug):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.cost_threshold = cost_threshold
        self._debug = debug

    def learn(self, weight_matrix, target_matrix, feature_matrix):
        probability_computer = ProbabilityComputerFactory.create_probability_computer("softmax")
        cost_computer = CostComputerFactory.create_cost_computer("neglog")
        for epoch in range(0, self.epoch):
            probability_matrix = probability_computer.compute_probability(np.dot(feature_matrix, weight_matrix))
            cost = cost_computer.compute_cost(target_matrix, probability_matrix)
            if self._debug:
                print cost
            weight_matrix = UpdateWeights.update_weights(weight_matrix, probability_matrix, target_matrix,
                                                         feature_matrix, self.learning_rate)
            if cost < self.cost_threshold:
                return weight_matrix
        return weight_matrix


class LearningProcessFactory(object):
    @staticmethod
    def create_learning_process(learning_rate, epoch, cost_threshold, debug):
        return Learn(learning_rate, epoch, cost_threshold, debug)
