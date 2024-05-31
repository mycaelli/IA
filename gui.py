import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

def load_data():
    '''
    Função que carrega os dados para validação dos caracteres
    Garante que os dados estejam em uma única dimensão para que possam ser usados no treinamento
    '''
    input = np.load('./content/X.npy')  # dados vêm no formato de matriz
    output_data = np.load('./content/Y_classe.npy')
    input_data = []
    for d in input:
        character = d.flatten()
        input_data.append(character)
    return input_data, output_data

def load_logic_gate_data():
    '''
    Função que carrega os dados para as portas lógicas AND, OR e XOR
    '''
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    Y_and = np.array([[0], [0], [0], [1]])
    Y_or = np.array([[0], [1], [1], [1]])
    Y_xor = np.array([[0], [1], [1], [0]])

    return X, Y_and, Y_or, Y_xor

class MLP:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate=0.01):
        '''
        Inicialização dos pesos (aleatória e uniformizada), bias e learning rate da rede
        '''
        self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, [input_layer_size + 1, hidden_layer_size])  # +1 para o bias
        self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, [hidden_layer_size + 1, output_layer_size])  # +1 para o bias
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        'função sigmoide que é utilizada como função de ativação'
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        'derivada da função de ativação'
        return x * (1 - x)

    def feedforward(self, X):
        'função que proporciona o resultado do treinamento da rede'
        X_bias = np.insert(X, 0, 1, axis=1)  # Adiciona bias
        self.hidden_input = np.dot(X_bias, self.weights_input_to_hidden)
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.hidden_output_bias = np.insert(self.hidden_output, 0, 1, axis=1)  # Adiciona bias
        self.final_input = np.dot(self.hidden_output_bias, self.weights_hidden_to_output)
        output = self.sigmoid(self.final_input)
        return output

    def backpropagation(self, X, Y, output):
        'função que atualiza os pesos da rede de acordo com os erros encontrados nos resultados obtidos pelo feedforward'
        X_bias = np.insert(X, 0, 1, axis=1)  # Adiciona bias
        self.hidden_output_bias = np.insert(self.hidden_output, 0, 1, axis=1)  # Adiciona bias

        output_error = Y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = np.dot(output_delta, self.weights_hidden_to_output.T[:, 1:])
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_to_output += self.learning_rate * np.dot(self.hidden_output_bias.T, output_delta)
        self.weights_input_to_hidden += self.learning_rate * np.dot(X_bias.T, hidden_delta)

    def train(self, X_train, Y_train, X_val, Y_val, epochs=1000):
        accuracies = []
        errors = []
        val_accuracies = []
        val_errors = []

        for epoch in range(epochs):
            output = self.feedforward(X_train)
            self.backpropagation(X_train, Y_train, output)

            error = np.mean(np.square(Y_train - output))
            errors.append(error)

            accuracy = accuracy_score(Y_train.argmax(axis=1), np.round(output).argmax(axis=1))
            accuracies.append(accuracy)

            # Validação
            val_output = self.feedforward(X_val)
            val_error = np.mean(np.square(Y_val - val_output))
            val_errors.append(val_error)
            val_accuracy = accuracy_score(Y_val.argmax(axis=1), np.round(val_output).argmax(axis=1))
            val_accuracies.append(val_accuracy)

        return accuracies, errors, val_accuracies, val_errors

# Carregar os dados dos caracteres
X, Y = load_data()
X = np.array(X)

# Dividir os dados em treino, validação e teste
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Treinar o MLP
mlp = MLP(input_layer_size=120, hidden_layer_size=3, output_layer_size=26, learning_rate=0.04)
initial_weights = {'weights_input_to_hidden': mlp.weights_input_to_hidden, 'weights_hidden_to_output': mlp.weights_hidden_to_output}
accuracies, errors, val_accuracies, val_errors = mlp.train(X_train, Y_train, X_val, Y_val, epochs=1000)
final_weights = {'weights_input_to_hidden': mlp.weights_input_to_hidden, 'weights_hidden_to_output': mlp.weights_hidden_to_output}

print("Initial Weights:")
print(initial_weights)
print("\nFinal Weights:")
print(final_weights)

# Plotar as acurácias de treinamento e validação
plt.figure(figsize=(10, 6))
plt.plot(accuracies, label='Acurácia de Treinamento')
plt.plot(val_accuracies, label='Acurácia de Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.title('Acurácia ao longo das Épocas')
plt.legend()
plt.show()

# Plotar os erros de treinamento e validação
plt.figure(figsize=(10, 6))
plt.plot(errors, label='Erro de Treinamento')
plt.plot(val_errors, label='Erro de Validação')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.title('Erro de Treinamento e Validação ao longo das Épocas')
plt.legend()
plt.show()

# Carregar e testar dados das portas lógicas
X_logic, Y_and, Y_or, Y_xor = load_logic_gate_data()

# Ajustar o MLP para problemas lógicos (2 entradas, 1 saída)
mlp_logic = MLP(input_layer_size=2, hidden_layer_size=3, output_layer_size=1, learning_rate=0.1)

print("Treinando para porta AND")
accuracies_and, errors_and, _, _ = mlp_logic.train(X_logic, Y_and, X_logic, Y_and, epochs=10000)
print("Resultados porta AND:", mlp_logic.feedforward(X_logic))

print("Treinando para porta OR")
accuracies_or, errors_or, _, _ = mlp_logic.train(X_logic, Y_or, X_logic, Y_or, epochs=10000)
print("Resultados porta OR:", mlp_logic.feedforward(X_logic))

print("Treinando para porta XOR")
accuracies_xor, errors_xor, _, _ = mlp_logic.train(X_logic, Y_xor, X_logic, Y_xor, epochs=10000)
print("Resultados porta XOR:", mlp_logic.feedforward(X_logic))