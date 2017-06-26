This code downloaded from https://github.com/Hvass-Labs/TensorFlow-Tutorials.

Image classification model is trained to classifiy the CIFAR-10 dataset. 
The maximum training accuary achieved was 96.1%, however, testing accuracy on average was 83%. 

The various network model were tried but six layered network resulted good accuracy. After 32,000 training iteration on CPU core i5, the summary is given below.   
Min: 50.8

Max: 96.1

Mean: 82.29

Std: 8.96

Processing Time: 44 hours (i5 4GB RAM)

The network model selected has two CNN layers of 96 neurons and two of 192 neurons, followed by two fully connected layer of 256 and 128 activation units. Thus making total of six layers. The conv Net layers is followed by polling with kernel size (3,3) and stride of two. The graph of training accuracy is given below:

https://github.com/mohbattharani/cifar10/blob/master/graph.jpg

By increasing depth of network and training iteration we can get better results.

