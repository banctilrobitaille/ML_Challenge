from dataInstance import DataInstance


class Dataset:
    __dataInstances = []
    __flattenDataArray = None

    def __init__(self, flatten_data_array=None, images=None):
        if flatten_data_array:
            self.__flattenDataArray = flatten_data_array
        elif images:
            self.__dataInstances = map(lambda image: DataInstance.from_image(image), images)
    
    @property
    def flattenDataArray(self):
        return self.__flattenDataArray

    @flattenDataArray.setter
    def flattenDataArray(self, flatten_data_array):
        self.__flattenDataArray = flatten_data_array
