# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        

        self._X = X
   
        self._y = one_hot_encoding(y)

        np.random.seed(42)

        # The random weights are the bias are initialized and multiploed with 0.1 to normalise


        self._h_weights = np.random.randn(np.shape(X)[1],self.n_hidden) * 0.1
        self._o_weights = np.random.randn(self.n_hidden,np.shape(self._y)[1]) * 0.1


        self._h_bias = np.random.randn(1,self.n_hidden) * 0.1
        self._o_bias = np.random.randn(1,np.shape(self._y)[1]) * 0.1


    # Function to perform forward propagation
    
    def forward_propagation(self):

        QQ1 = self._X.dot(self._h_weights)  + self._h_bias
        SS1 = self.hidden_activation(QQ1)


        QQ2 = SS1.dot(self._o_weights) + self._o_bias
        SS2 = self._output_activation(QQ2)

        return QQ1,SS1,QQ2,SS2

    # Function to perform backward propogation


    def back_propagation(self,A1,A2,Z1):

        der_z2 = A2 - self._y
        der_w2 = (1 / self._y.size) * np.matmul(np.transpose(A1),der_z2)
        der_b2 = (1 / self._y.size) * np.sum(der_z2)

        der_z1 = np.matmul(der_z2,np.transpose(self._o_weights)) * self.hidden_activation(Z1,True)
        der_w1 = (1 / self._y.size) * np.matmul(np.transpose(self._X),der_z1)
        der_b1 = (1 / self._y.size) * np.sum(der_z1)

        return der_w1,der_b1,der_w2,der_b2

        # Function to perform gradient descent

    def gradient_descent(self,der_w1 ,der_b1, der_w2, der_b2):

        self._h_weights = self._h_weights - (self.learning_rate * der_w1) 
        self._h_bias = self._h_bias - (self.learning_rate * der_b1) 
        
        self._o_weights  = self._o_weights  - (self.learning_rate * der_w2)
        self._o_bias = self._o_bias - (self.learning_rate * der_b2)

    # The equation of the above 3 functions are refferred from the image in https://www.kaggle.com/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras


    # Function to fit the model

    def fit(self, X, y):

        self._initialize(X, y)

        for i in range(self.n_iterations):

            # Forward propagation is called

            QQ1,SS1,QQ2,SS2 = self.forward_propagation()

            # After 20 iteration the loss function si calculated

            if(i%20 == 0):

                self._loss_history.append(self._loss_function(self._y,SS2))


            der_w1,der_b1,der_w2,der_b2 = self.back_propagation(SS1,SS2,QQ1)


            self.gradient_descent(der_w1 ,der_b1, der_w2, der_b2)


    # FUnction to predict the data

    def predict(self, X):

        pred = []
        

        QQ1 = X.dot(self._h_weights)  + self._h_bias
        SS1 = self.hidden_activation(QQ1)


        QQ2 = SS1.dot(self._o_weights) + self._o_bias
        SS2 = self._output_activation(QQ2)

        
        for i in SS2:
            pred.append(np.argmax(i))

        return pred
        
