class Dataset(object):
    __data_instances = []

    def with_data_instances(self, data_instances):
        self.__data_instances = data_instances
        return self

    @property
    def data_instances(self):
        return self.__data_instances

    @data_instances.setter
    def data_instances(self, data_instances):
        self.__data_instances = data_instances
