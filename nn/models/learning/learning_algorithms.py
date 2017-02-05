import abc
import numpy as np
from utils.label_mapper import LabelMapper
from datetime import datetime


class LearningAlgorithmTypes(object):
    SGD = "stochastic gradient descent"


class LearningAlgorithmFactory(object):
    @staticmethod
    def create_learning_algorithm_from_type(learning_algorithm_type):
        if learning_algorithm_type == LearningAlgorithmTypes.SGD:
            return SGD()
        else:
            raise NotImplementedError("Requested learning algorithm type not yet implemented")


class AbstractLearningAlgorithm(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        raise NotImplementedError()


class SGD(AbstractLearningAlgorithm):
    def __init__(self):
        super(SGD, self).__init__()

    def learn(self, network, training_data_set, number_of_epochs, learning_rate, size_of_batch, **kwargs):
        for epoch in range(number_of_epochs):
            print("Epoch: " + str(epoch) + " Start time: " + str(datetime.now()))
            np.random.shuffle(training_data_set.data_instances)

            for batch in np.array_split(training_data_set.data_instances,
                                        len(training_data_set.data_instances) / size_of_batch):
                self.__update_weights_and_bias(network, batch, learning_rate)

    def __update_weights_and_bias(self, network, batch, learning_rate):
        number_of_training_instances = len(batch)
        updated_biases = map(lambda layer_biases: np.zeros(layer_biases.shape), network.biases)
        updated_weights = map(lambda layer_weights: np.zeros(layer_weights.shape), network.weights)

        for data_instance in batch:
            # Computing the partial derivatives of the function cost w.r.t. each weight and bias. These partial
            # derivatives are the gradient's components.
            delta_biases, delta_weights = self.__back_propagate(network, data_instance.features_values_vector,
                                                                data_instance.label)

            # Accumulating the delta of weights and biases for each training sample of the batch in order to adjust
            # the network's weights and biases in the opposite direction of the gradient.
            updated_biases = [new_bias + delta for new_bias, delta in zip(updated_biases, delta_biases)]
            updated_weights = [new_weight + delta for new_weight, delta in zip(updated_weights, delta_weights)]

        # Updating the network's weights and biases in the opposite direction of the cost function's gradient
        network.weights = [current_weight - (learning_rate / number_of_training_instances) * new_weight
                           for current_weight, new_weight in zip(network.weights, updated_weights)]
        network.biases = [current_bias - (learning_rate / number_of_training_instances) * new_bias
                          for current_bias, new_bias in zip(network.biases, updated_biases)]

    def __back_propagate(self, network, output_vector, expected_output_label):
        last_layer = -1
        updated_biases = map(lambda layer_biases: np.zeros(layer_biases.shape), network.biases)
        updated_weights = map(lambda layer_weights: np.zeros(layer_weights.shape), network.weights)

        output_vectors_by_layer = [output_vector.reshape(1, len(output_vector))]
        input_vectors_by_layer = []

        for bias, weights in zip(network.biases, network.weights):
            next_layer_input = np.dot(output_vector, weights) + bias.T
            input_vectors_by_layer.append(next_layer_input)
            output_vector = network.neuron.compute(next_layer_input)
            output_vectors_by_layer.append(output_vector)

        delta = network.cost_computer.compute_cost_derivative(output_vector=output_vectors_by_layer[last_layer],
                                                              expected_output_vector=LabelMapper().map_label_to_vector(
                                                                      expected_output_label)) * \
                network.neuron.compute_derivative(input_vectors_by_layer[last_layer])
        updated_biases[last_layer] = delta.T
        updated_weights[last_layer] = np.dot(output_vectors_by_layer[last_layer - 1].T, delta)

        for layer_index in xrange(2, network.number_of_layers):
            z = input_vectors_by_layer[-layer_index]
            sp = network.neuron.compute_derivative(z)
            delta = np.dot(delta, network.weights[-layer_index + 1].T) * sp
            updated_biases[-layer_index] = delta.T
            updated_weights[-layer_index] = np.dot(output_vectors_by_layer[-layer_index - 1].T, delta)

        return updated_biases, updated_weights
