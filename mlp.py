import sys
sys.path.append('../')

import ia.function as function
import ia.matrix as matrix
import ia.metrics.metrics as metrics
import ia.visualization.plot as plot

import numpy as np
# Implementação da MLP
class MLP:

    def __init__(self, input_size, hidden_size, output_size, learning_rate):

        self.hidden_weights = matrix.initialize_matrix(input_size, hidden_size)
        print(self.hidden_weights)
        self.output_weights = matrix.initialize_matrix(hidden_size, output_size)
        print(self.output_weights)
        self.hidden_bias = 0
        self.output_bias = 0
        self.learning_rate = learning_rate # de 0 a 1

    def feedforward(self, X):

        hidden_input_layer_aux = matrix.matmul(X, self.hidden_weights)
        input_hidden_layer = matrix.add(hidden_input_layer_aux, self.hidden_bias)
    
        output_hidden_layer = function.sigmoid(input_hidden_layer)

        input_exit_layer_aux = matrix.matmul(output_hidden_layer, self.output_weights)
        input_exit_layer = matrix.add(input_exit_layer_aux, self.output_bias)

        predicted_output = function.sigmoid(input_exit_layer) # saida do neuronio

        return predicted_output, input_exit_layer
    
    def backpropagate(self, target, predicted_output, output_hidden_layer):

        error_aux = matrix.subtract(target, predicted_output)
        derive_predicted_output = function.sigmoid_derivative(predicted_output)
        error_info = matrix.multiply(error_aux, derive_predicted_output)

        weight_adjustment_aux = matrix.multiply(self.learning_rate, error_info)
        weight_adjustment = matrix.multiply(weight_adjustment_aux, output_hidden_layer)

        bias_adjustment = matrix.multiply(self.learning_rate, error_info)

        return weight_adjustment, bias_adjustment
    
    def update_weights(self, weight_adjustment, bias_adjustment):

        weight_adjustment_transpose = matrix.transpose(weight_adjustment)
        self.hidden_weights = matrix.add(self.hidden_weights, weight_adjustment_transpose)
        self.hidden_bias = matrix.add(self.hidden_bias, bias_adjustment)
        self.output_weights = matrix.add(self.output_weights, weight_adjustment)
        self.output_bias = matrix.add(self.output_bias, bias_adjustment)

    '''''
        X: input data
        y: expected output
    '''
    def train(self, input_data, expected_output, epochs):

        # loss_values = []
        # accuracy_values = []
        # precision_values = []
        # recall_values = []

        for epoch in range(epochs):
            # self.learning_rate = self.learning_rate + 0.01
            # aprendizado
            print("running...")
            print('Epoch: ',epoch)
            old_hidden_weights = self.hidden_weights.copy()
            old_output_weights = self.output_weights.copy()
            predicted_output, input_exit_layer  = self.feedforward(input_data)  # Passagem para frente
            weight_adjustment, bias_adjustment = self.backpropagate(expected_output, predicted_output, input_exit_layer)  # Passagem para trás
            self.update_weights(weight_adjustment, bias_adjustment)

            
            print('predicted_output')
            # print(expected_output)
            print()
            print(predicted_output)

            # metrica
            # loss, accuracy, precision, recall = metrics.calculate_metrics(predicted_output, expected_output)
            # print('Epoch: ',epoch, ':' , loss, accuracy, precision, recall)
            print('-----------------------------')
            # loss_values.append(loss)
            # accuracy_values.append(accuracy)
            # precision_values.append(precision)
            # recall_values.append(recall)
            
            
            if (old_hidden_weights == self.hidden_weights).all() and  (old_output_weights == self.output_weights).all():
                break

        # plot.plot_loss(loss_values, epoch, self.learning_rate)
        # plot.plot_metrics(accuracy_values, precision_values, recall_values, epoch, self.learning_rate)
        # plot.plot_confusion_matrix(predicted_output, expected_output, epoch, self.learning_rate)
