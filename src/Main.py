import Data as dat
import numpy as np
import NeuralNetwork as nn

#create a matrix with each row containint 3 random numbers and a solution to those numbers with some equation
def generate_output(input):    
    return 3 * (input[0]**2 * input[4]) + (5 * input[1] * input[2] * input[3]) + np.log(input[2] * input[5]) + 2 * input [6] * input[7] + input[8]**input[9] 


num_inputs = 10

#Main
train_x, train_y = dat.create_data(num_inputs, 20000, generate_output)


#print(test_x)
print("*********************************")
#print(test_y)


net = nn.NeuralNetwork([train_x.shape[0], 9, 6, 5, 4, 4, 3, train_y.shape[0]], nn.Logger(True, 2))

for i in range(1, 11):
    test_x, test_y = dat.create_data(num_inputs, 300, generate_output)
    predictions = net.predict(test_x)
    total = test_y.shape[1]
    matches = 0
    for n in range(0, test_y.shape[1]):
        if test_y[0, n] == predictions[0, n]:
            matches = matches + 1

    print(f'Before training: #{i} success = {matches/total}')

print("*********************************")

net.train(train_x, train_y, 10, 0.0001)

for i in range(1, 11):
    test_x, test_y = dat.create_data(num_inputs, 300, generate_output)
    predictions = net.predict(test_x)
    total = test_y.shape[1]
    matches = 0
    for n in range(0, test_y.shape[1]):
        if test_y[0, n] == predictions[0, n]:
            matches = matches + 1

    print(f'Success #{i}= {matches/total}')
print("*********************************")