import unittest
import numpy as np
from mockito import when, mock
from core.network import FeedForwardNetwork


class LearningAlgorithmsTests(unittest.TestCase):
    __feed_forward_network_mock = None
    __input_vector = np.array([1, 2, 3, 4, 5])
    __

    def setUp(self):
        self.__feed_forward_network_mock = mock(FeedForwardNetwork)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
