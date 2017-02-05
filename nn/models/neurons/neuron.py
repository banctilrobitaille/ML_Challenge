import abc
import numpy as np


class NeuronTypes(object):
    SIGMOID = "sigmoid"


class AbstractNeuron(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute(self, vector):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_derivative(self, vector):
        raise NotImplementedError()


class SigmoidNeuron(AbstractNeuron):
    def compute(self, vector):
        return 1.0 / (1.0 + np.exp(-vector))

    def compute_derivative(self, vector):
        return self.compute(vector) * (1 - self.compute(vector))


class NeuronFactory(object):
    @staticmethod
    def create_neuron_from_type(neuron_type):
        if neuron_type == NeuronTypes.SIGMOID:
            return SigmoidNeuron()
