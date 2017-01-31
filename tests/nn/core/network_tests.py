import unittest
import numpy as np

from core.network import NetworkFactory, NetworkTypes
from models.learning.learning_algorithms import LearningAlgorithmTypes
from models.neurons.neuron import NeuronTypes
from models.cost_computers.cost_computer import CostFunctionTypes


class FeedForwardNetworkTest(unittest.TestCase):
    __feedForward_network = None

    def setUp(self):
        self.__feedForward_network = NetworkFactory.create_network_with(network_type=NetworkTypes.FEED_FORWARD,
                                                                        number_of_layers=3,
                                                                        number_of_neurons_per_layer=[3, 4, 2],
                                                                        type_of_neuron=NeuronTypes.SIGMOID,
                                                                        cost_function_type=CostFunctionTypes.QUADRATIC,
                                                                        learning_algorithm_type=LearningAlgorithmTypes.SGD)

    def tearDown(self):
        pass

    def test_should_compute_output_vector_when_forwarding_input(self):
        input_vector = np.array([1, 2, 3])
        expected_output_vector = np.array([[0.99996464, 0.99996464]])

        # Creating biases
        first_layer_biases = np.array([[0.25], [0.25], [0.25], [0.25]])
        first_layer_biases.reshape(4, 1)
        second_layer_biases = np.array([[0.25], [0.25]])
        second_layer_biases.reshape(2, 1)

        # Creating weights
        first_layer_weights = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
        second_layer_weights = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

        self.__feedForward_network.weights = [first_layer_weights, second_layer_weights]
        self.__feedForward_network.biases = [first_layer_biases, second_layer_biases]
        np.testing.assert_almost_equal(self.__feedForward_network.accept(input_vector), expected_output_vector)

        if __name__ == '__main__':
            unittest.main()
