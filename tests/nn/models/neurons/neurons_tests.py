import unittest

from nn.models.neurons.neuron import SigmoidNeuron


class NeuronsTests(unittest.TestCase):
    __sigmoid_neuron = None

    def setUp(self):
        self.__sigmoid_neuron = SigmoidNeuron()

    def tearDown(self):
        pass

    def test_should_compute_sigmoid(self):
        input_value = 5
        expected_output = 0.993307149075

        self.assertAlmostEqual(self.__sigmoid_neuron.compute(input_value), expected_output)


if __name__ == "__main__":
    unittest.main()
