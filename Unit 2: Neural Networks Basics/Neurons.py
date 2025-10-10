#############################################################
# Neurons module
# 
# Author: Dr. Jared McBride (10-9-2025, Buena Vista, VA)
# 
# This file contains the array of 4 different artificial neurons.
# They were developed in class as part a unit of neural networks 
# basics.
# 
# These are based of the perceptron and adiline models from Raschka
# from his book "Python Machine Learning" (3rd edition). As well as
# the perceptron model found in David McKay's book "Information
# Theory, Inference and Learning Algorithms".
#
# The file is organized as follows:
# 1. Import libraries
# 2. Neuron classes
#    2.1 Perceptron (Raschka)
#    2.2 Adaline
#    2.3 Adaline Stochastic Gradient Descent
#    2.4 Adaline Mini-Batch Gradient Descent
#    2.5 Perceptron (McKay)
#
#############################################################

import numpy as np

class Perceptron(object):

    def __init__(self, eta=0.01, n_epochs=20, random_seed=1):
        self.eta = eta
        self.n_epochs = n_epochs
        self.random_seed = random_seed

    def fit(self, X, Y):
        # Initializing ieghts
        # Start by setting the random seed
        random_generator = np.random.RandomState(self.random_seed)

        # initialize weights
        self.w_ = random_generator.normal(loc=0,scale=0.01,size= 1 + X.shape[1])

        # Initialize errors list
        self.errors_ = []

        for j in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X,Y):
                yhat = self.predict(xi)
                error = target - yhat
                dw = self.eta * error * xi
                dw = np.r_[self.eta * error , dw] # add bias weight update

                # Apply learning rule
                self.w_ += dw

                errors += int((target - self.predict(xi)) != 0.0)

            self.errors_.append(errors)

        return self
    
    def predict(self, xi):
        z = self.w_ @ np.r_[[1], xi] # activity
        yhat = np.where(z >= 0, 1, -1)
        return yhat
    
#############################################################

class Adaline(object):

    def __init__(self, eta=0.01, n_epochs=20, random_seed=1):
        self.eta = eta
        self.n_epochs = n_epochs
        self.random_seed = random_seed

    def fit(self, X, Y):
        # Initializing ieghts
        # Start by setting the random seed
        random_generator = np.random.RandomState(self.random_seed)

        # initialize weights
        self.w_ = random_generator.normal(loc=0,scale=0.01,size= 1 + X.shape[1])

        # Initialize (sum of squares) errors list
        self.cost_ = []

        for j in range(self.n_epochs):
            # Y - phi(z) NOT the final prediction which includes thresholding
            errors = Y - (np.dot(X, self.w_[1:]) + self.w_[0]) # The whole vector of targets minus predictions of all X
            
            # errors is n by 1 , X is n by p (number features)
            dw = self.eta * np.array([sum(errors * X[:,j]) for j in range(X.shape[1])])

            # dw then is p by 1 this line line below makes it (p+1) by 1. The first element is for the bias
            dw = np.r_[self.eta * sum(errors) , dw] 

            self.w_ += dw

            cost = sum(errors**2) / 2.0
            self.cost_.append(cost)
        return self
    
    def predict(self, X):
        z = self.w_ @ np.c_[np.ones(X.shape[0]), X].T # Activity
        # Adaline uses f(z) = z as activation then thresholds after
        yhat = np.where(z >= 0, 1, -1) # Predictions made by thresholding
        return yhat
    

#############################################################

class AdalineSGD(object):

    def __init__(self, eta=0.01, n_epochs=20, shuffle = True, random_seed=1):
        self.eta = eta
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.random_seed = random_seed

    def fit(self, X, Y):
        # Initializing ieghts
        # Start by setting the random seed
        random_generator = np.random.RandomState(self.random_seed)

        # initialize weights
        self.w_ = random_generator.normal(loc=0,scale=0.01,size= 1 + X.shape[1])

        # Initialize (sum of squares) errors list
        self.cost_ = []

        for j in range(self.n_epochs):

            # Shuffle the data at the beginning of each epoch is desired
            if self.shuffle:
                idx = np.arange(X.shape[0])
                random_generator.shuffle(idx)
                X = X[idx]
                Y = Y[idx]

            cost = 0

            for xi, target in zip(X,Y):
                activity = np.dot(xi, self.w_[1:]) + self.w_[0]
                y_tilde = self.activation(activity) 

                # y_tilde is y_hat before thresholding
                error = target - y_tilde
            
                # errors is a number (based on a single record), xi is 1 by p (number features), a single record
                dw = self.eta * error * xi

                # dw then is p by 1 this line line below makes it (p+1) by 1. The first element is for the bias
                dw = np.r_[self.eta * error , dw] 

                self.w_ += dw

                # Accumulate cost
                cost += error**2
            self.cost_.append(cost/2.0)
        return self
    
    def predict(self, X):
        z = self.w_ @ np.c_[np.ones(X.shape[0]), X].T # Activity
        # Adaline uses f(z) = z as activation then thresholds after
        yhat = np.where(z >= 0, 1, -1) # Predictions made by thresholding
        return yhat
    
    def activation(self, z):
        return z
    
#############################################################

class AdalineMBGD(object):

    def __init__(self, eta=0.01, n_epochs=20, shuffle = True, random_seed=1):
        self.eta = eta
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.random_seed = random_seed

    def fit(self, X, Y):
        # Initializing ieghts
        # Start by setting the random seed
        random_generator = np.random.RandomState(self.random_seed)

        # initialize weights
        self.w_ = random_generator.normal(loc=0,scale=0.01,size= 1 + X.shape[1])

        # Initialize (sum of squares) errors list
        self.cost_ = []

        for j in range(self.n_epochs):

            # Shuffle the data at the beginning of each epoch is desired
            if self.shuffle:
                idx = np.arange(X.shape[0])
                random_generator.shuffle(idx)
                X = X[idx]
                Y = Y[idx]

            cost = 0

            for start in range(0, X.shape[0], 10): # Mini-batch size of 10
                end = start + 10
                X_mini = X[start:end,:]
                Y_mini = Y[start:end]

                errors = Y_mini - (np.dot(X_mini, self.w_[1:]) + self.w_[0]) # The whole vector of targets minus predictions of all X
            
                # errors is n by 1 , X is n by p (number features)
                dw = self.eta * np.array([sum(errors * X_mini[:,j]) for j in range(X_mini.shape[1])])

                # dw then is p by 1 this line line below makes it (p+1) by 1. The first element is for the bias
                dw = np.r_[self.eta * sum(errors) , dw] 

                self.w_ += dw

                cost = sum(errors**2) / 2.0
            self.cost_.append(cost)
        return self
    
    def predict(self, X):
        z = self.w_ @ np.c_[np.ones(X.shape[0]), X].T # Activity
        # Adaline uses f(z) = z as activation then thresholds after
        yhat = np.where(z >= 0, 1, -1) # Predictions made by thresholding
        return yhat
    
    def activation(self, z):
        return z
    


#############################################################

class PerceptronMcKay(object):
    
    def __init__(self, eta=0.01, n_epochs=20, random_seed=1):
        self.eta = eta
        self.n_epochs = n_epochs
        self.random_seed = random_seed

    def fit(self, X, Y):
        # Initializing ieghts
        # Start by setting the random seed
        random_generator = np.random.RandomState(self.random_seed)

        # initialize weights
        self.w_ = random_generator.normal(loc=0,scale=0.01,size= 1 + X.shape[1])

        # Initialize errors list
        self.errors_ = []

        for j in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X,Y):

                yhat = self.predict(xi)
                error = target - yhat

                # Update weights
                dw = self.eta * error * xi
                self.w_ += np.r_[self.eta * error , dw] 

                errors += error**2 / 2.0

            self.errors_.append(errors)

        return self
    
    def predict(self, xi):
        z = self.w_ @ np.r_[1, xi] # activity
        yhat = self.activation(z)
        return yhat
    
    def activation(self, z):
        # Logistic sigmoid activation function
        return 1.0/(1.0 + np.exp(-z))