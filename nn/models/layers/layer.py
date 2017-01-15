from nn.models.neurons.neuron import SigmoidNeuron

class Layer(object):
    __id = 0
    __neurons = []
    __next_layer = None
    __previous_layer = None

    def __init__(self, layer_id, number_of_neurons):
        self.__id = layer_id
        self.__neurons = map(lambda neuron: SigmoidNeuron(), range(0, number_of_neurons))
