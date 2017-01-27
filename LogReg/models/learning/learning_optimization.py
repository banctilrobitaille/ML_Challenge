import numpy as np
from LogReg.models.feature_computers.prediction_computer import ProbabilityComputerFactory
from LogReg.models.cost_computers.cost_computer import CostComputerFactory


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


class Learn(object):
    def __init__(self, learnig_rate=0.1, epoch=100, cost_threshold=0.1, debug=False):
        self.learning_rate = learnig_rate
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
            weight_matrix -= UpdateWeights.update_weights(weight_matrix, probability_matrix, target_matrix,
                                                          feature_matrix, self.learning_rate)
            if cost < self.cost_threshold:
                return weight_matrix
        return weight_matrix
