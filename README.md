# ML Challenge <img src="/robot.png" width="60" vertical-align="bottom"> [![Build Status](https://travis-ci.org/ADozois/ML_Challenge.svg?branch=master)](https://travis-ci.org/ADozois/ML_Challenge )
> This machine learning challenge has been done as part of the Machine Learning(IN 2064) lecture at Technische Universität München. It basically consists of implementing four machine learning algorithms (K Nearest Neighbors, Neural Network, Log Regression and Gaussian Process) in order to classify the well known MNIST data set that can be downloaded following this link: http://yann.lecun.com/exdb/mnist/. The four algorithms have to classify handwritten digits with minimum error. Data pre processing was authorized.

### K Nearest Neighbors
The K Nearest Neighbors (K-NN) can be a very efficient and easy to implement algorithm to classify data. But, with a large data set like MNIST (60000 samples in the training set and a test set of 10000 images) it takes a very long time to compute. This is due to the fact we have to compute the distance (eucledean distance for instance) of every test sample against every training sample in order to find the K nearest neighbors and hopefully correctly classify our data. To accelerate the process, we've implemented a multiprocessing K-NN. 

#### Usage example:
  <img src="/images/multiknn.png">
#### Preliminary Results
Even with multiprocessing K-NN it took more than 4 minutes to classify 100 samples.
##### Global success rate : 97% 
  <img src="/images/result_knn_2.png">
  
#### How to contribute ?
- [X] Create a branch by feature and/or bug fix
- [X] Get the code
- [X] Commit and push
- [X] Create a pull request

#### Branch naming

##### Feature branch
> feature/ [Short feature description] [Issue number]

#####Bug branch
> fix/ [Short fix description] [Issue number]

#### Commits syntax:

#####Adding code:
> \+ Added [Short Description] [Issue Number]

#####Deleting code:
> \- Deleted [Short Description] [Issue Number]

#####Modifying code:
> \* Changed [Short Description] [Issue Number]


Icons made by <a href="http://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="http://www.flaticon.com" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a>
