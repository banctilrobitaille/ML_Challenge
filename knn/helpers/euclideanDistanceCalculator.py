from scipy.spatial.distance import euclidean


class EuclideanDistanceCalculator(object):
    @staticmethod
    def calculate_distance_between(data_instance1, data_instance2):
        return euclidean(data_instance1.features_values_vector, data_instance2.features_values_vector)
