import numpy as np
import abc


class CostFunctionTypes(object):
    NEGLOG = "neglog"


class CostFunction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_cost(self, *inputs):
        raise NotImplementedError


class NegLogLikelihoodCostComputer(CostFunction):
    @classmethod
    def compute_cost(cls, target_matrix, probability_matrix):
        return (-(np.log(probability_matrix) * target_matrix).sum(1)).mean()


class CostComputerFactory(object):
    @staticmethod
    def create_cost_computer(cost_function_type):
        if cost_function_type == CostFunctionTypes.NEGLOG:
            return NegLogLikelihoodCostComputer()
