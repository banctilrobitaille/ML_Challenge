import numpy as np
import abc

from models.cost_computers.cost_computer import CostComputerFactory
from models.learning.learning_algorithms import LearningAlgorithmFactory
from models.neurons.neuron import NeuronFactory


class NetworkTypes(object):
    FEED_FORWARD = "feed forward"


class NetworkFactory(object):
    @staticmethod
    def create_network_with(network_type, number_of_layers, number_of_neurons_per_layer, type_of_neuron,
                            cost_function_type, learning_algorithm_type):
        if network_type == NetworkTypes.FEED_FORWARD:
            return FeedForwardNetwork(number_of_layers, number_of_neurons_per_layer, type_of_neuron,
                                      cost_function_type, learning_algorithm_type)
        else:
            raise NotImplementedError("Requested neural network type not yet implemented")


class AbstractNetwork(object):
    __metaclass__ = abc.ABCMeta
    __number_of_layers = 0
    __biases = None
    __weights = None
    __neuron = None
    __cost_computer = None
    __learning_algorithm = None

    def __init__(self, number_of_layers, number_of_neurons_per_layer, type_of_neuron, cost_function_type,
                 learning_algorithm_type):
        self.__number_of_layers = number_of_layers
        self.__biases = [np.random.randn(y, 1) for y in number_of_neurons_per_layer[1:]]
        self.__weights = [np.random.randn(y, x) for x, y in
                          zip(number_of_neurons_per_layer[:-1], number_of_neurons_per_layer[1:])]

        self.__neuron = NeuronFactory.create_neuron_from_type(type_of_neuron)
        self.__cost_computer = CostComputerFactory.create_cost_computer_from_type(cost_function_type)
        self.__learning_algorithm = LearningAlgorithmFactory.create_learning_algorithm_from_type(
                learning_algorithm_type)

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

    @property
    def learning_algorithm(self):
        return self.__learning_algorithm

    @abc.abstractmethod
    def accept(self, input_vector):
        raise NotImplementedError()

    @abc.abstractmethod
    def learn(self, training_data_set, number_of_epochs, learning_rate, size_of_batch, **kwargs):
        raise NotImplementedError()


class FeedForwardNetwork(AbstractNetwork):
    def __init__(self, number_of_layers, number_of_neurons_per_layer, type_of_neuron, cost_function_type,
                 learning_algorithm_type):
        super(FeedForwardNetwork, self).__init__(number_of_layers, number_of_neurons_per_layer, type_of_neuron,
                                                 cost_function_type, learning_algorithm_type)

    def accept(self, input_vector):
        for biases, weights in zip(self.biases, self.weights):
            output_vector = self.neuron.compute(np.dot(weights, input_vector) + biases)
        return output_vector

    def learn(self, training_data_set, number_of_epochs, learning_rate, size_of_batch, **kwargs):
        self.learning_algorithm.learn(self, training_data_set, number_of_epochs, learning_rate, size_of_batch)
