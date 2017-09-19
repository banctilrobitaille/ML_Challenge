# ML Challenge <img src="/robot.png" width="60" vertical-align="bottom"> [![Build Status](https://travis-ci.org/ADozois/ML_Challenge.svg?branch=master)](https://travis-ci.org/ADozois/ML_Challenge )
> This machine learning challenge has been done as part of the Machine Learning(IN 2064) lecture at Technische Universität München. It basically consists of implementing four machine learning algorithms (K Nearest Neighbors, Neural Network, Log Regression and Gaussian Process) in order to classify the well known MNIST data set that can be downloaded following this link: http://yann.lecun.com/exdb/mnist/. The four algorithms have to classify handwritten digits with minimum error. Data pre processing was authorized.

### K Nearest Neighbors
The K Nearest Neighbors (K-NN) can be a very efficient and easy to implement algorithm to classify data. But, with a large data set like MNIST (60000 samples in the training set and a test set of 10000 images) it takes a very long time to compute. This is due to the fact we have to compute the distance (eucledean distance for instance) of every test sample against every training sample in order to find the K nearest neighbors and hopefully correctly classify our data. To accelerate the process, we've implemented a multiprocessing K-NN. 

#### Usage example:
```python
  from knn.core.classifier import KnnClassifier as knn
  from knn.core.classifier import MultiProcessedKnnClassifier as multi_processed_knn
  # Simple KNN
  knn(training_data_set=training_data_set).classify(data_set=test_data_set, number_of_neighbors=10)
  # Multi-processed KNN
  multi_processed_knn(training_data_set=training_data_set).classify(data_set=test_data_set, number_of_neighbors=10)
```
#### Preliminary Results
Even with multiprocessing K-NN it took more than 6 minutes to classify 1000 samples.
##### Global success rate : 94.1% with 49 features, and 95.3% with 196 features, both with Otsu thresholding as data pre process
  <img src="/images/result_knn_3.PNG">
  
### Feed Forward Neural Network

#### Usage example:
```python
  from nn.models.neurons.neuron import NeuronTypes
  from nn.models.cost_computers.cost_computer import CostFunctionTypes
  from nn.models.learning.learning_algorithms import LearningAlgorithmTypes
  from nn.core.network import NetworkFactory, NetworkTypes
  
  neural_network = NetworkFactory.create_network_with(network_type=NetworkTypes.FEED_FORWARD,
                                                        number_of_layers=4,
                                                        number_of_neurons_per_layer=[784, 50, 25, 10],
                                                        type_of_neuron=NeuronTypes.SIGMOID,
                                                        cost_function_type=CostFunctionTypes.QUADRATIC,
                                                        learning_algorithm_type=LearningAlgorithmTypes.SGD)
   neural_network.learn(training_data_set=training_data_set, number_of_epochs=75, learning_rate=0.5, size_of_batch=200)
   neural_network.classify(test_data_set)
```

#### Preliminary Results

##### Global success rate : 92.37% with a 4 layers feed forward neural network using SGD (Stochastic gradient descent) as learning algorithm.

<img src="/images/result_nn.PNG">

#### How to contribute ?
- [X] Create a branch by feature and/or bug fix
- [X] Get the code
- [X] Commit and push
- [X] Create a pull request

#### Branch naming

##### Feature branch
> feature/ [Short feature description] [Issue number]

##### Bug branch
> fix/ [Short fix description] [Issue number]

#### Commits syntax:

##### Adding code:
> \+ Added [Short Description] [Issue Number]

##### Deleting code:
> \- Deleted [Short Description] [Issue Number]

##### Modifying code:
> \* Changed [Short Description] [Issue Number]


Icons made by <a href="http://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="http://www.flaticon.com" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a>
