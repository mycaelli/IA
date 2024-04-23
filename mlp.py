import sys
sys.path.append('../')

import ia.function as function
import ia.matrix as matrix
import ia.metrics.metrics as metrics
import ia.visualization.plot as plot

# Implementação da MLP
class MLP:

    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        """Inicializa a rede neural com uma camada oculta."""
        # Inicializando pesos e bias
        self.weights_input_to_hidden = matrix.initialize_matrix(input_size, hidden_size)
        self.weights_hidden_to_output = matrix.initialize_matrix(hidden_size, output_size)
        self.bias_hidden = matrix.initialize_matrix(1, hidden_size)
        self.bias_output = matrix.initialize_matrix(1, output_size)
        self.learning_rate = learning_rate

    '''
        propaga os dados pela rede neural -> é o que da o output da rede

        X: input data

        P/ cada camada
        saída = F(Dados de entrada x Pesos) + Bias, onde F é a função de ativação
    '''
    def feedforward(self, X):
        """Executa uma passagem para frente na rede."""

        # Calcula a saída da camada oculta
        hidden_layer_input = matrix.array_product(X, self.weights_input_to_hidden) # + self.bias_hidden
        hidden_layer_output =  matrix.apply_function(function.sigmoid, hidden_layer_input)

        # Calcula a saida da camada de saida
        output_layer_input = matrix.array_product(hidden_layer_output, self.weights_hidden_to_output) # + self.bias_hidden
        predicted_output = matrix.apply_function(function.sigmoid, output_layer_input)

        return hidden_layer_output, predicted_output
    
    '''
        calcula os gradiented dos pesos da rede em relação ao erro da saida
        é o que permite que o erro seja minimizado ao longo do treinamento

        error = true_output - predicted_output
        delta = erro * derivada da funcao de ativacao
    '''
    def backpropagate(self, true_output, hidden_layer_output, predicted_output):
        """Executa o algoritmo de backpropagation para calcular os gradientes."""

        # Calcula o erro da camada de saída
        output_error = matrix.elementwise_subtraction(matrix.transpose(true_output), predicted_output)
        output_delta = output_error * function.sigmoid_derivative(predicted_output)
        
        # Calcula o erro da camada oculta
        hidden_error = matrix.array_product(output_delta, matrix.transpose(self.weights_hidden_to_output))
        hidden_delta = hidden_error * function.sigmoid_derivative(hidden_layer_output)

        return output_delta, hidden_delta
    
    # entender essa funcao
    def update_weights(self, X, output_delta, hidden_delta, hidden_layer_output):

        self.weights_hidden_to_output += matrix.array_product(matrix.transpose(hidden_layer_output), output_delta) * self.learning_rate
        self.bias_output += matrix.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_to_hidden += matrix.array_product(matrix.transpose(X), hidden_delta) * self.learning_rate
        self.bias_hidden += matrix.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    '''''
        X: input data
        y: expected output

    '''
    def train(self, input_data, expected_output, epochs):
        loss_values = []
        accuracy_values = []
        precision_values = []
        recall_values = []
        for epoch in range(epochs):
            hidden_layer_output, predicted_output = self.feedforward(input_data)  # Passagem para frente
            loss, accuracy, precision, recall = metrics.calculate_metrics(predicted_output, expected_output)
            loss_values.append(loss), accuracy_values.append(accuracy), precision_values.append(precision), recall_values.append(recall)
            if (expected_output == predicted_output).all(): # VERIFICAR SE PODE USAR ALL AQUI, é numpy
                print('UHUUUU')
            else:
                output_delta, hidden_delta = self.backpropagate(matrix.transpose(expected_output), hidden_layer_output, predicted_output)  # Passagem para trás
                self.update_weights(input_data, output_delta, hidden_delta, hidden_layer_output)
        plot.plot_loss(loss_values, epoch, self.learning_rate), plot.plot_metrics(accuracy_values, precision_values, recall_values, epoch, self.learning_rate)