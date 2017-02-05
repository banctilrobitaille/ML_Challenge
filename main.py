from commons.exceptions.unableToLoadDatasetException import UnableToLoadDatasetException
from commons.exceptions.unableToSaveDatasetException import UnableToSaveDatasetException
from commons.helpers.fileHelper import FileHelper
from commons.models.constants.classificationMethod import ClassificationMethod
from commons.models.constants.datasetType import DatasetType
from commons.models.datasetFactory import DatasetFactory
from nn.models.cost_computers.cost_computer import CostFunctionTypes
from nn.core.network import NetworkFactory, NetworkTypes
from nn.models.neurons.neuron import NeuronTypes
from nn.models.learning.learning_algorithms import LearningAlgorithmTypes
from knn.core.classifier import KnnClassifier as knn
from knn.core.classifier import MultiProcessedKnnClassifier as multi_processed_knn
from logreg.core.loreg import LogRegClassifier
import sklearn.gaussian_process.gaussian_process

def launch_knn_classification_with(number_of_instances_to_classify):
    try:
        training_data_set, test_data_set = FileHelper.load_data_sets_for(ClassificationMethod.KNN)
    except UnableToLoadDatasetException as e:
        print(e.message)
        training_data_set, test_data_set = DatasetFactory.create_and_save_data_set_from_files(
                with_data_normalization=False,
                with_threshold=True,
                number_of_features=196,
                classification_method=ClassificationMethod.KNN)

    if number_of_instances_to_classify >= 1000:
        multi_processed_knn(training_data_set=training_data_set).classify(data_set=test_data_set,
                                                                          number_of_neighbors=10)
    else:
        knn(training_data_set=training_data_set).classify(data_set=test_data_set, number_of_neighbors=10)


def launch_neural_network_classification():
    try:
        training_data_set, test_data_set = FileHelper.load_data_sets_for(ClassificationMethod.NN)
    except UnableToLoadDatasetException as e:
        print(e.message)
        training_data_set, test_data_set = DatasetFactory.create_and_save_data_set_from_files(
                with_data_normalization=True,
                with_threshold=False,
                number_of_features=784,
                classification_method=ClassificationMethod.NN)

    neural_network = NetworkFactory.create_network_with(network_type=NetworkTypes.FEED_FORWARD,
                                                        number_of_layers=4,
                                                        number_of_neurons_per_layer=[784, 50, 25, 10],
                                                        type_of_neuron=NeuronTypes.SIGMOID,
                                                        cost_function_type=CostFunctionTypes.QUADRATIC,
                                                        learning_algorithm_type=LearningAlgorithmTypes.SGD)
    neural_network.learn(training_data_set=training_data_set, number_of_epochs=75, learning_rate=0.5, size_of_batch=200)
    neural_network.classify(test_data_set)
    FileHelper.save_trained_model(neural_network, ClassificationMethod.NN)


def launch_log_reg_classification():
    try:
        training_data_set, test_data_set = FileHelper.load_data_sets_for(ClassificationMethod.KNN)
    except UnableToLoadDatasetException as e:
        print(e.message)
        training_data_set, test_data_set = DatasetFactory.create_and_save_data_set_from_files(
                with_data_normalization=False,
                with_threshold=True,
                number_of_features=196,
                classification_method=ClassificationMethod.LOG_REG)

    log_reg = LogRegClassifier(training_data_set, 0.1)
    log_reg.train(number_epoch=10000, cost_threshold=0.25, debug=True)
    log_reg.classify(test_data_set)


if __name__ == '__main__':
    # launch_knn_classification_with(1)
    # launch_neural_network_classification()
    launch_log_reg_classification()