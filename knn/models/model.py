from knn.helpers.euclideanDistanceCalculator import EuclideanDistanceCalculator
from knn.models.neighborhood import Neighborhood


class Model(object):
    __training_dataset = None

    def train_with_dataset(self, dataset):
        self.__training_dataset = dataset
        return self

    def classify(self, data_instance_to_classify, number_of_neighbors):
        neighborhood = Neighborhood(number_of_neighbors=number_of_neighbors)

        for data_instance in self.__training_dataset.data_instances:
            distance = EuclideanDistanceCalculator.calculate_distance_between(data_instance_to_classify, data_instance)
            neighborhood.accept(data_instance, distance)

        return neighborhood.get_the_label_trend()
