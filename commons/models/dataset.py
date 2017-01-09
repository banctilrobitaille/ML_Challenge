class Dataset:
    __flattenDataArray = None

    def __init__(self, flatten_data_array=None):
        if flatten_data_array:
            self.__flattenDataArray = flatten_data_array

    @property
    def flattenDataArray(self):
        return self.__flattenDataArray

    @flattenDataArray.setter
    def flattenDataArray(self, flatten_data_array):
        self.__flattenDataArray = flatten_data_array
