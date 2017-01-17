import abc


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

    def learn(self, network, training_data_set, learning_rate, number_of_epochs, size_of_batch, **kwargs):
        for epoch in range(number_of_epochs):
            pass
