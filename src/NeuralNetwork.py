import numpy as np
from Logger import *

class NeuralNetwork:    

    def __init__(self, nn_definition, logger = Logger(True, 1)):
        self._nn_definition = nn_definition
        self._logger = logger
        self._parameters = self.initialze_parameters()

    def train(self, input_data, output_data, num_iterations = 500, learning_rate = 0.05):
        num_layers = len(self._nn_definition) #layers including input layer       
        for iter in range(0, num_iterations):
            outputs = self._forward_propagation(input_data, self._parameters)
            self._logger.log(f'Outputs: {outputs}', 1)
            cost = self._calculate_cost(outputs[f'A{num_layers-1}'], output_data)

            if iter % 50 == 0:
                self._logger.log(f'Cost = {cost} for iteration {iter}', 2)            
            backs = self._back_propagation(self._parameters, outputs, input_data, output_data)
     
            for n in range(1, num_layers):
                self._parameters[f'W{n}'] = self._parameters[f'W{n}'] - learning_rate * backs[f'W{n}']
                self._parameters[f'b{n}'] = self._parameters[f'b{n}'] - learning_rate * backs[f'b{n}']
     
    def predict(self, input_data):
        o = len(self._nn_definition) - 1
        outputs = self._forward_propagation(input_data, self._parameters)
        return np.vectorize(lambda i: 1 if i >= 0.5 else 0)(outputs[f'A{o}'])

    def initialze_parameters(self):
        self._logger.log(f'Initializing neural network with shape {self._nn_definition}', 2)
        num_layers = len(self._nn_definition) #layers including input layer
        parameters = {}
        
        for n in range(1, num_layers):
            parameters[f'W{n}'] = np.random.randn(self._nn_definition[n], self._nn_definition[n-1])
            parameters[f'b{n}'] = np.zeros((self._nn_definition[n], 1))      

        self._logger.log(f'Initialized parameters: {parameters}', 1)
        return parameters

    def _forward_propagation(self, X, parameters):
        num_layers = len(self._nn_definition) #layers including input layer        
        output = {}
        A_prev = X
        for n in range(1, num_layers - 1):
            output[f'Z{n}'] = np.dot(parameters[f'W{n}'], A_prev) + parameters[f'b{n}']
            output[f'A{n}'] = self._relu(output[f'Z{n}'])
            A_prev = output[f'A{n}']

        n_last = num_layers - 1
        output[f'Z{n_last}'] = np.dot(parameters[f'W{n_last}'], A_prev) + parameters[f'b{n_last}']
        output[f'A{n_last}'] = self._simgoid(output[f'Z{n_last}'])    

        return output

    def _calculate_cost(self, prediction, actual): # (1, num_training_examples)
        m = prediction.shape[1]

        ly = np.where(prediction != 0., np.log(prediction), 0)
        lly = np.where((1 - prediction) != 0., np.log(1- prediction), 0)

        cost = -1/m * np.sum( np.multiply(actual, ly) + np.multiply( (1 - actual), lly) )
       
        #cost = 1/m * np.sum(actual - self._normalize(prediction, num_classes)) ** 2, axis = 1)
        return cost        

    def _back_propagation(self, parameters, outputs, input_data, output_data):
        num_layers = len(self._nn_definition) #layers including input layer        
        m = output_data.shape[1]

        backs = {}

        backs[f'Z{num_layers-1}'] = outputs[f'A{num_layers-1}'] - output_data
        backs[f'W{num_layers-1}'] = 1/m * np.dot(backs[f'Z{num_layers-1}'], outputs[f'A{num_layers-2}'].T)
        backs[f'b{num_layers-1}'] = 1/m * np.sum(backs[f'Z{num_layers-1}'], axis=1, keepdims=True)        

        for n in range(num_layers-2, 0, -1):
            backs[f'Z{n}'] = np.dot(parameters[f'W{n+1}'].T, backs[f'Z{n+1}']) * self._relu_prime(outputs[f'Z{n}'])
            backs[f'W{n}'] = 1/m * np.dot(backs[f'Z{n}'], outputs[f'A{n-1}'].T if n > 1 else input_data.T )
            backs[f'b{n}'] = 1/m * np.sum(backs[f'Z{n}'], axis=1, keepdims=True)
        return backs

    def _relu(self, n):
        return np.vectorize(lambda i: i if i > 0 else 0)(n)

    def _relu_prime(self, n):
        return np.vectorize(lambda i: 1 if i > 0 else 0)(n)

    def _simgoid(self, n):        
        s = 1/(1+np.exp(-n))
        return np.vectorize(lambda i: 1 if i == float("inf") else 0 if i == -float("inf") or i == float("nan") else i)(s)
            

    #y has values from 0 to 1, we want to turn that into values from 0 to num_classes-1
    #def _de_normalize(self, y, num_classes):
    #    return np.vectorize(lambda i: int(num_classes - 1) if i == 1 else int(i * num_classes))(y)

    #y has n classes represented as integers from 0 to n-1.
    #normalize it to be a number from 0 to 1
    #def _normalize(self, y, num_classes):
    #    return np.vectorize(lambda i: 1 if i == num_classes - 1 else i/float(num_classes))(y)