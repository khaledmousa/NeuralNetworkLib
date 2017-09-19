import numpy as np

#Creates simple test data for ML training
# num_of_inputs: number of input features
# num_of_examples: numer of examples to generate
# generator: a function that receives a row with random input data and generates a single output value
# num_classes: the number of output values, the outputs from generator are normalized to discreet values from 0 to num_classes-1
def create_data(num_of_inputs, num_of_examples, generator, num_classes=2):
    X = np.random.randint(1, 100, (num_of_inputs, num_of_examples))
    Y = np.zeros((1, num_of_examples))    
    for n in range(0, num_of_examples):      
        Y[0, n] = generator(X[:,n]) #generate_output(X[:,n])
    class_boud = (np.max(Y)+1)/num_classes
    Y = np.vectorize(lambda y: int(y/class_boud), otypes=[int])(Y)
    return X, Y
