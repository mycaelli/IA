import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_data():
    input_path = 'C:\\Users\\gigic\\Downloads\\IA-main\\IA-main\\content\\X.npy'
    output_path = 'C:\\Users\\gigic\\Downloads\\IA-main\\IA-main\\content\\Y_classe.npy'

    input_data = np.load(input_path)
    output_data = np.load(output_path)

    input_data = input_data.reshape(input_data.shape[0], -1)

    return input_data, output_data


class MLP:
    def __init__(self, input_layer_size, hidden_layer_sizes, output_layer_size, learning_rate=0.01):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.weights = []
        self.biases = []

        # Inicialização dos pesos e biases
        previous_layer_size = input_layer_size + 1
        for hidden_layer_size in hidden_layer_sizes:
            self.weights.append(np.random.uniform(-0.5, 0.5, [previous_layer_size, hidden_layer_size]))
            self.biases.append(np.full(hidden_layer_size, 1, dtype=float))
            previous_layer_size = hidden_layer_size + 1

        self.weights.append(np.random.uniform(-0.5, 0.5, [previous_layer_size, output_layer_size]))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        activations = [np.insert(X, 0, 1, axis=1)]
        for i in range(len(self.hidden_layer_sizes)):
            weighted_sum = np.dot(activations[-1], self.weights[i])
            activation = self.sigmoid(weighted_sum)
            activations.append(np.insert(activation, 0, 1, axis=1))

        final_weighted_sum = np.dot(activations[-1], self.weights[-1])
        output = self.sigmoid(final_weighted_sum)

        return output, activations

    def backpropagation(self, X, Y, output, activations):
        output_error = Y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        deltas = [output_delta]
        for i in reversed(range(len(self.hidden_layer_sizes))):
            error = np.dot(deltas[0], self.weights[i + 1].T[:, 1:])
            delta = error * self.sigmoid_derivative(activations[i + 1][:, 1:])
            deltas.insert(0, delta)

        self.weights[-1] += self.learning_rate * np.dot(activations[-1].T, output_delta)
        for i in range(len(self.hidden_layer_sizes) - 1, -1, -1):
            self.weights[i] += self.learning_rate * np.dot(activations[i].T, deltas[i])

    def train(self, X_train, Y_train, X_val, Y_val, epochs=1000):
        accuracies = []
        errors = []
        val_accuracies = []
        val_errors = []

        for epoch in range(epochs):
            output, activations = self.feedforward(X_train)
            self.backpropagation(X_train, Y_train, output, activations)

            error = np.mean(np.square(Y_train - output))
            errors.append(error)

            accuracy = accuracy_score(Y_train.argmax(axis=1), np.round(output).argmax(axis=1))
            accuracies.append(accuracy)

            # Validação
            val_output, _ = self.feedforward(X_val)
            val_error = np.mean(np.square(Y_val - val_output))
            val_errors.append(val_error)
            val_accuracy = accuracy_score(Y_val.argmax(axis=1), np.round(val_output).argmax(axis=1))
            val_accuracies.append(val_accuracy)

        return accuracies, errors, val_accuracies, val_errors


# Carregar os dados dos caracteres
X, Y = load_data()

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir os dados em treino, validação e teste
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Treinar o MLP com ajustes
mlp = MLP(input_layer_size=120, hidden_layer_sizes=[20, 10], output_layer_size=26, learning_rate=0.01)
initial_weights = {'weights_input_to_hidden': mlp.weights[0], 'weights_hidden_to_output': mlp.weights[-1]}
accuracies, errors, val_accuracies, val_errors = mlp.train(X_train, Y_train, X_val, Y_val, epochs=2000)
final_weights = {'weights_input_to_hidden': mlp.weights[0], 'weights_hidden_to_output': mlp.weights[-1]}

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
