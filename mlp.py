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
        self.hidden_weights = matrix.initialize_matrix(input_size, hidden_size)
        self.output_weights = matrix.initialize_matrix(hidden_size, output_size)
        # self.hidden_bias = matrix.initialize_matrix(1, hidden_size)
        # self.output_bias = matrix.initialize_matrix(1, output_size)
        self.learning_rate = learning_rate # de 0 a 1

    '''
        propaga os dados pela rede neural -> é o que da o output da rede

        X: input data

        P/ cada camada
        saída = F(Dados de entrada x Pesos) + Bias, onde F é a função de ativação

        z_1 = np.dot(X, W_1) # multiplica dados de entrada com pesos da primeira camada

        h_1 = 1 / (1 + np.exp(-z_1)) # produz os valores de ativação da camada oculta

        y_hat = np.dot(h_1, W_2) # multiplica os valores de ativacao da camada oculta pelos pesos da camada de saida

        L = np.sum(y_hat**2) # loss: calcula a funcao de perda entre as previsões y_hat e os rótulos reais (L contém o valor da função de perda)
    '''
    def feedforward(self, X):
        """Executa uma passagem para frente na rede."""

        hidden_input = matrix.array_product(X, self.hidden_weights)

        hidden_output = function.sigmoid(hidden_input)

        exit_input = matrix.array_product(hidden_output, self.output_weights)

        exit_prediction = function.sigmoid(exit_input)

        return exit_prediction, hidden_output
    
    '''
        calcula os gradiented dos pesos da rede em relação ao erro da saida
        é o que permite que o erro seja minimizado ao longo do treinamento

        error = true_output - predicted_output
        delta = erro * derivada da funcao de ativacao

        dy_hat = 2.0 * y_hat # calcula o gradiente da função de perda entre as previsões y_hat (o gradiente é 2 * erro)

        dW2 = h_1.T.dot(dy_hat) # calcula o gradiente da função de perda em relação aos pesos da camada de saida, onde h_1 = valores de ativação da camada oculta e dy_hat é o gradiente da função de perda

        dh1 = dy_hat.dot(W_2.T) # calcula o gradiente da função de perda em relação aos valores de ativação da camada oculta (h_1), onde W_2 é o peso da camada de saída

        dz1 = dh1 * h_1 * (1 - h_1) # Calcula o gradiente em relação aos valores de entrada da camada oculta z_1, usando a derivada da função sigmoid

        dW1 = X.T.dot(dz1) # calcula o gradiente da função de perda em relação aos pesos da primeira camada W_1, onde dz1 são os valores de entrada da camada oculta
    '''
    def backpropagate(self, exit_prediction, hidden_output, input_data):
        """Executa o algoritmo de backpropagation para calcular os gradientes."""

        gradient_loss_prediction = 2.0 * exit_prediction

        gradient_output_layer_weights = matrix.array_product(matrix.transpose(hidden_output), gradient_loss_prediction)

        gradient_activation_hidden_layer = matrix.array_product(gradient_loss_prediction, matrix.transpose(self.output_weights))

        gradient_input_hidden_layer = gradient_activation_hidden_layer * function.sigmoid_derivative(hidden_output)

        gradient_hidden_layer_weights = matrix.array_product(matrix.transpose(input_data), gradient_input_hidden_layer)



        return gradient_hidden_layer_weights, gradient_output_layer_weights
    
    '''
        W1 -= alpha * dW1 # W1 pesos da camada de entrada a camada oculta
        W2 -= alpha * dW2 # W2 pesos da camada oculta a saida
    '''
    def update_weights(self, gradient_hidden_layer_weights, gradient_output_layer_weights):


        self.hidden_weights -= gradient_hidden_layer_weights * self.learning_rate
        self.output_weights -= gradient_output_layer_weights * self.learning_rate
        # fazer pro bias


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
            # aprendizado
            old_hidden_weights = self.hidden_weights.copy()
            old_output_weights = self.output_weights.copy()
            output_prediction, hidden_output  = self.feedforward(input_data)  # Passagem para frente
            g_h_layer_weights, g_o_layer_weights = self.backpropagate(output_prediction, hidden_output, input_data)  # Passagem para trás
            self.update_weights(g_h_layer_weights,  g_o_layer_weights)

            
            print('expected_output, output_prediction')
            print(expected_output)
            print(output_prediction)
            # metrica
            loss, accuracy, precision, recall = metrics.calculate_metrics(output_prediction, expected_output)
            print('Epoch: ',epoch, ':' , loss, accuracy, precision, recall)
            print('-----------------------------')
            loss_values.append(loss)
            accuracy_values.append(accuracy)
            precision_values.append(precision)
            recall_values.append(recall)
            
            
            if (old_hidden_weights == self.hidden_weights).all() and  (old_output_weights == self.output_weights).all():
                break
        plot.plot_loss(loss_values, epoch, self.learning_rate)
        plot.plot_metrics(accuracy_values, precision_values, recall_values, epoch, self.learning_rate)
        plot.plot_confusion_matrix(output_prediction, expected_output, epoch, self.learning_rate)