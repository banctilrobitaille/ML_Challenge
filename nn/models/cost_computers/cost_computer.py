import abc


class CostFunctionTypes(object):
    QUADRATIC = "quadratic"
    CROSS_ENTROPY = "cross_entropy"


class CostComputer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_cost(self, *args):
        raise NotImplementedError()


class QuadraticCostComputer(CostComputer):
    def compute_cost(self, *args):
        pass


class CrossEntropyComputer(CostComputer):
    def compute_cost(self, *args):
        pass


class CostComputerFactory(object):
    @staticmethod
    def create_cost_computer_from_type(cost_function_type):
        if cost_function_type == CostFunctionTypes.QUADRATIC:
            return QuadraticCostComputer()
        elif cost_function_type == CostFunctionTypes.CROSS_ENTROPY:
            return CrossEntropyComputer()
