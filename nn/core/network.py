from nn.models.layers.layer import Layer


class Network(object):
    __layers = []

    def __init__(self, number_of_layer, number_of_neurons_per_layer=[]):
        self.__layers = map(lambda layer_id, number_of_neurons: Layer(layer_id, number_of_neurons),
                            range(0, number_of_layer), number_of_neurons_per_layer)
