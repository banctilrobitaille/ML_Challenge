import abc
import numpy as np


class CostFunctionTypes(object):
    QUADRATIC = "quadratic"
    CROSS_ENTROPY = "cross_entropy"


class CostComputer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_cost(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_cost_derivative(self, *args, **kwargs):
        raise NotImplementedError()


class QuadraticCostComputer(CostComputer):
    def compute_cost(self, *args, **kwargs):
        return (1.0 / kwargs['total_number_of_training_inputs']) * np.square(
                np.linalg.norm(kwargs['approximated_output_vector'] - kwargs['expected_output_vector']))

    def compute_cost_derivative(self, *args, **kwargs):
        return kwargs["output_vector"] - kwargs["expected_output_vector"]


class CrossEntropyComputer(CostComputer):
    def compute_cost(self, *args, **kwargs):
        pass

    def compute_cost_derivative(self, *args, **kwargs):
        pass


class CostComputerFactory(object):
    @staticmethod
    def create_cost_computer_from_type(cost_function_type):
        if cost_function_type == CostFunctionTypes.QUADRATIC:
            return QuadraticCostComputer()
        elif cost_function_type == CostFunctionTypes.CROSS_ENTROPY:
            return CrossEntropyComputer()
