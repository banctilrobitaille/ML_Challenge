import numpy as np

class DataAggregator(object):

    @staticmethod
    def aggregate_to_class(data):
        list_=[]
        list_of_list=[]
        for number in range(0,10):
            instance = np.array(data.get_instances_with_label(number))
            list_.append(instance)
            list_of_list.append(list_)
        data = np.array(list_of_list)
        return data