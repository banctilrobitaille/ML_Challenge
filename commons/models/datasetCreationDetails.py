class InstanceCreationDetails(object):
    __with_threshold = None
    __with_normalization = None
    __number_of_features = 0

    def __init__(self, with_threshold, with_normalization, number_of_features):
        self.__with_threshold = with_threshold
        self.__with_normalization = with_normalization
        self.__number_of_features = number_of_features

    @property
    def with_threshold(self):
        return self.__with_threshold

    @property
    def with_normalization(self):
        return self.__with_normalization

    @property
    def number_of_features(self):
        return self.__number_of_features
