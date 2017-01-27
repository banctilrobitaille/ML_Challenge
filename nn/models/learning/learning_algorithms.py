import abc
import numpy as np
from commons.helpers.datasetHelper import DatasetHelper


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
            np.random.shuffle(training_data_set.data_instances)

            for batch in np.array_split(training_data_set.data_instances,
                                        len(training_data_set.data_instances) / size_of_batch):
                self.__update(network, batch, learning_rate)

    def __update(self, network, batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in network.biases]
        nabla_w = [np.zeros(w.shape) for w in network.weights]

        for data_instance in batch:
            delta_nabla_b, delta_nabla_w = self.back_propagate(network, DatasetHelper.normalize(
                data_instance.features_values_vector),
                                                               data_instance.label)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        network.weights = [w - (learning_rate / len(batch)) * nw
                           for w, nw in zip(network.weights, nabla_w)]
        network.biases = [b - (learning_rate / len(batch)) * nb
                          for b, nb in zip(network.biases, nabla_b)]

    def back_propagate(self, network, x, y):
        nabla_b = [np.zeros(b.shape) for b in network.biases]
        nabla_w = [np.zeros(w.shape) for w in network.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(network.biases, network.weights):
            z = np.dot(activation, w) + b
            zs.append(z)
            activation = network.neuron.compute(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * network.neuron.compute_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, network.number_of_layers):
            z = zs[-l]
            sp = network.neuron.compute(z)
            delta = np.dot(network.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        return output_activations - y
