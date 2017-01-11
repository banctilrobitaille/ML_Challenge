class DataInstance:
    __features = {}
    __label = None

    def __init__(self):
        pass

    @property
    def features(self):
        return self.__features

    @features.setter
    def features(self, features):
        self.__features = features

    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, label):
        self.__label = label
