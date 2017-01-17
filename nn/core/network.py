import numpy as np
import abc

from models.cost_computers.cost_computer import CostComputerFactory
from models.neurons.neuron import NeuronFactory


class NetworkTypes(object):
    REGULAR = "regular"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"


class NetworkFactory(object):
    @staticmethod
    def create_network_with(network_type, number_of_layers, number_of_neurons_per_layer, type_of_neuron,
                            cost_function_type):
        if network_type == NetworkTypes.REGULAR:
            return RegularNetwork(number_of_layers, number_of_neurons_per_layer, type_of_neuron,
                                  cost_function_type)
        else:
            raise NotImplementedError()


class AbstractNetwork(object):
    __metaclass__ = abc.ABCMeta
    __number_of_layers = 0
    __biases = None
    __weights = None
    __neuron = None
    __cost_computer = None

    def __init__(self, number_of_layers, number_of_neurons_per_layer, type_of_neuron, cost_function_type):
        self.__number_of_layers = number_of_layers
        self.__biases = [np.random.randn(y, 1) for y in number_of_neurons_per_layer[1:]]
        self.__weights = [np.random.randn(y, x) for x, y in
                          zip(number_of_neurons_per_layer[:-1], number_of_neurons_per_layer[1:])]

        self.__neuron = NeuronFactory.create_neuron_from_type(type_of_neuron)
        self.__cost_computer = CostComputerFactory.create_cost_computer_from_type(cost_function_type)

    @property
    def biases(self):
        return self.__biases

    @biases.setter
    def biases(self, biases):
        self.__biases = biases

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        self.__weights = weights

    @property
    def neuron(self):
        return self.__neuron

    @property
    def number_of_layers(self):
        return self.__number_of_layers

    @abc.abstractmethod
    def feed_forward(self, input_vector):
        raise NotImplementedError()


class RegularNetwork(AbstractNetwork):
    def __init__(self, number_of_layers, number_of_neurons_per_layer, type_of_neuron, cost_function_type):
        super(RegularNetwork, self).__init__(number_of_layers, number_of_neurons_per_layer, type_of_neuron,
                                             cost_function_type)

    def feed_forward(self, values_vector):
        for biases, weights in zip(self.biases, self.weights):
            values_vector = self.neuron.compute(np.dot(weights, values_vector) + biases)
        return values_vector
