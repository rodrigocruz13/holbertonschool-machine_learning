#!/usr/bin/env python3
"""
Class DeepNeuralNetwork
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle


class DeepNeuralNetwork:
    """ Class """

    def __init__(self, nx, layers):
        """
        Initialize NeuralNetwork
        Args:
            - nx: nx is the number of input features
            - Layers: is the number of nodes found in the hidden layer
        Public attributes:
            - L: The number of layers in the neural network.
            - cache: A dictionary to hold all intermediary values of the
            network. Upon instantiation, it should be set to an empty dict.
            - weights: A dict to hold all weights and biased of the network.
            Upon instantiation:
            - The weights of the network should be initialized with He et al.
            method and saved in the weights dictionary using the key W{l}
            where {l}is the hidden layer the weight belongs to
            - The biases of the network should be initialized to 0’s and
            saved in the weights dictionary using the key b{l} where {l}
            is the hidden layer the bias belongs to
        """

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')

        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        # Privatizing attributes

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lay in range(len(layers)):
            if not isinstance(layers[lay], int) or layers[lay] <= 0:
                raise TypeError('layers must be a list of positive integers')

            self.weights["b" + str(lay + 1)] = np.zeros((layers[lay], 1))

            if lay == 0:
                sq = np.sqrt(2 / nx)
                he_et_al = np.random.randn(layers[lay], nx) * sq
                self.weights["W" + str(lay + 1)] = he_et_al

            else:
                sq = np.sqrt(2 / layers[lay - 1])
                he_et_al = np.random.randn(layers[lay], layers[lay - 1]) * sq
                self.weights["W" + str(lay + 1)] = he_et_al

    @property
    def L(self):
        """
        getter method
        Args:
            - self
        Return:
        - __L: NUmber of layers .
        """
        return self.__L

    @property
    def cache(self):
        """
        getter method
        Args:
            - self
        Return:
        - __cache: (dict) Has al intermediaty values of the network.
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter method
        Args:
            - self
        Return:
        - __weights: (dict) Has al the weights and bias of the network.
        """
        return self.__weights


    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Updates the private attributes __cache
        The activated outputs of each layer should be saved in the __cache
        dictionary using the key A{l} where {l} is the hidden layer the
        activated output belongs to
        The neurons should use a sigmoid activation function


        Arguments:
        - X is a numpy.ndarray with shape (nx, m) that contains the input data
          - nx (int) is the number of input features to the neuron
          - m (int) is the number of examples
        Return:
        - The output of the neural network and the cache, respectively
        """

        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):

            w = self.__weights["W" + str(layer)]
            a = self.__cache["A" + str(layer - 1)]
            b = self.__weights["b" + str(layer)]

            # z1 = w . X1 + b1
            z = np.matmul(w, a) + b

            # sigmoid function
            if (layer == self.__L):
                e = np.exp(z)
                forward_prop = e / np.sum(e, axis=0, keepdims=True)
            else:
                e = np.exp(-1 * z)
                forward_prop = 1 / (1 + e)

            # updating cache
            self.__cache["A" + str(layer)] = forward_prop

        return (forward_prop, self.__cache)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        - A is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        Return:
        - cost: the cost
        Answer from: https://bit.ly/37x9YzM
        """
        m = Y.shape[1]
        cost = -1 * (1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
         - X is a numpy.ndarray with shape (nx, m) & contains the input data

        - Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        Return:
        - Prediction: The prediction should be a numpy.ndarray with shape
                      (1, m) containing the predicted labels for each
                      example. The label values should be 1 if the output
                      of the network is >= 0.5 and 0 otherwise
        - cost: the cost
        Answer from: https://bit.ly/37x9YzM
        """

        # Generate forward propagation.
        labels = one_hot_encode(Y.T, (Y.max() + 1))

        self.forward_prop(X)
        A = self.__cache['A' + str(self.__L)]

        evaluate_predict = np.where(A < 0.5, 0, 1)
        cost = self.cost(labels, A)

        return (evaluate_predict, cost)


    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Arguments
        - Y       : numpy.ndarray
                    Array with shape (1, m) that contains the correct labels
                    for the input data
        - cache   : dictionary
                    Dictionary containing all the intermediary values of the
                    network
        - alpha   : learning rate

        Returns
            Updated the private attribute __weights
        """

        m = Y.shape[1]
        cp_w = self.__weights.copy()
        la = self.__L
        dz = self.__cache['A' + str(la)] - Y
        dw = np.dot(self.__cache['A' + str(la - 1)], dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        self.__weights['W' + str(la)] = cp_w['W' + str(la)] - alpha * dw.T
        self.__weights['b' + str(la)] = cp_w['b' + str(la)] - alpha * db

        for la in range(self.__L - 1, 0, -1):
            g = self.__cache['A' + str(la)] * (1 - self.__cache['A' + str(la)])
            dz = np.dot(cp_w['W' + str(la + 1)].T, dz) * g
            dw = np.dot(self.__cache['A' + str(la - 1)], dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            self.__weights['W' + str(la)] = cp_w['W' + str(la)] - alpha * dw.T
            self.__weights['b' + str(la)] = cp_w['b' + str(la)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network and Updates the private attributes
        __weights and __cache

        Arguments
        ---------
        - X         : numpy.ndarray
                      Array with shape (nx, m) that contains the input data
             nx     : int
                      number of input features to the neuron
             m      : int
                      the number of examples
        - Y         : numpy.ndarray
                      Array with shape (1, m) that contains the correct labels
                      for the input data
        - iterations: int
                      number of iterations to train over
        - alpha     : float
                      learning rate
        - verbose   : bool
                      Defines whether or not to print info about the training
        - graph     : bool
                      Defines whether or not to graph info about the training

        Returns
        - ev        : Float
                      The evaluation of the training data after iterations of
                      training have occurred
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")

        if (iterations < 0):
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if (alpha < 0):
            raise ValueError("alpha must be positive")

        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError('step must be an integer')

            if (step < 1) or (step > iterations):
                raise ValueError('step must be positive and <= iterations')

        steps = []
        costs = []

        for cont in range(iterations + 1):
            self.forward_prop(X)
            cache = self.__cache
            self.gradient_descent(Y, cache, alpha)

            if cont == iterations or cont % step == 0:
                cost = self.cost(Y, self.__cache['A' + str(self.__L)])

                if verbose:
                    print('Cost after {} iterations: {}'.format(cont, cost))

                if graph:
                    costs.append(cost)
                    steps.append(cont)

        if graph:
            fig = plt.figure(figsize=(10, 10))

            plt.plot(steps, costs, linewidth=3, markevery=10)
            plt.title('Training Cost')
            plt.ylabel('cost')
            plt.xlabel('iteration')
            fig.set_facecolor("white")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Arguments
        ---------
        - flename         : str
                            string with the name to use for the file. filename
                            is the file to which the object should be saved.
                            If filename doesn't have the extension .pkl, add it

        Returns
        -------
        Nothing
        """

        filename = filename + ".pkl" if ('.pkl'not in filename) else filename

        try:
            file_binary = open(filename, 'wb')
            pickle.dump(self, file_binary)
            file_binary.close()

        except BaseException:
            return None

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Arguments
        ---------
        - flename         : str
                            string with the name to use for the file. filename
                            is the file to which the object should be loaded.
                            If filename doesn't got the extension .pkl, add it

        Returns
        -------
        The loaded object, or None if filename doesn’t exist
        """

        try:
            DeepNeuralNetwork_file = open(filename, 'rb')
            info = pickle.load(DeepNeuralNetwork_file)
            DeepNeuralNetwork_file.close()
            return info

        except BaseException:
            return None


    def one_hot_encode(Y, classes):
        """
        Converts a numeric label vector into a one-hot matrix:
        Args:
            - Y is a numpy.ndarray with shape (m, )
            containing numeric class labels
            - m is the number of examples

        Returns:
            a one-hot encoding of Y with shape (classes, m),
            None on failure
        """

        if not isinstance(Y, np.ndarray):
            return None

        if len(Y) == 0:
            return None

        if (not isinstance(classes, int)):
            return None

        if (classes <= Y.max()):
            return None

        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        colum = np.arange(m)
        one_hot[Y, colum] = 1

        return one_hot
