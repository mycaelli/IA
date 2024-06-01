import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

def load_data():
    input_path = 'C:\\Users\\gigic\\Downloads\\IA-main\\IA-main\\content\\X.npy'
    output_path = 'C:\\Users\\gigic\\Downloads\\IA-main\\IA-main\\content\\Y_classe.npy'

    input_data = np.load(input_path)
    output_data = np.load(output_path)

    input_data = input_data.reshape(input_data.shape[0], -1)

    return input_data, output_data

def load_logic_gate_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    Y = Y.reshape(-1, 1)
    return X, Y

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

    def predict(self, X):
        output, _ = self.feedforward(X)
        return output

# Criar o diretório de saída, se não existir
output_dir = 'C:\\Users\\gigic\\Downloads\\IA-main\\IA-main\\arquivos de saida'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carregar os dados dos caracteres
X, Y = load_data()

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Configurar validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

# Listas para armazenar os resultados de cada fold
all_accuracies = []
all_errors = []
all_val_accuracies = []
all_val_errors = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]

    # Treinar o MLP com ajustes
    mlp = MLP(input_layer_size=120, hidden_layer_sizes=[20, 10], output_layer_size=26, learning_rate=0.01)
    initial_weights = {'weights_input_to_hidden': mlp.weights[0], 'weights_hidden_to_output': mlp.weights[-1]}
    accuracies, errors, val_accuracies, val_errors = mlp.train(X_train, Y_train, X_val, Y_val, epochs=2000)
    final_weights = {'weights_input_to_hidden': mlp.weights[0], 'weights_hidden_to_output': mlp.weights[-1]}

    print(f"Fold {fold} Initial Weights:")
    print(initial_weights)
    print(f"\nFold {fold} Final Weights:")
    print(final_weights)

    # Plotar as acurácias de treinamento e validação
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, label=f'Acurácia de Treinamento - Fold {fold}')
    plt.plot(val_accuracies, label=f'Acurácia de Validação - Fold {fold}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title(f'Acurácia ao longo das Épocas - Fold {fold}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"accuracy_plot_fold_{fold}.png"))
    plt.show()

    # Plotar os erros de treinamento e validação
    plt.figure(figsize=(10, 6))
    plt.plot(errors, label=f'Erro de Treinamento - Fold {fold}')
    plt.plot(val_errors, label=f'Erro de Validação - Fold {fold}')
    plt.xlabel('Épocas')
    plt.ylabel('Erro')
    plt.title(f'Erro de Treinamento e Validação ao longo das Épocas - Fold {fold}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"error_plot_fold_{fold}.png"))
    plt.show()

    # Armazenar os resultados de cada fold
    all_accuracies.append(accuracies)
    all_errors.append(errors)
    all_val_accuracies.append(val_accuracies)
    all_val_errors.append(val_errors)

    # Salvar arquivos de saída para cada fold
    np.save(os.path.join(output_dir, f"initial_weights_fold_{fold}.npy"), initial_weights)
    np.save(os.path.join(output_dir, f"final_weights_fold_{fold}.npy"), final_weights)
    np.save(os.path.join(output_dir, f"train_errors_fold_{fold}.npy"), errors)
    np.save(os.path.join(output_dir, f"val_errors_fold_{fold}.npy"), val_errors)

    fold += 1

# Carregar e validar dados das portas lógicas
logic_gate_files = ['problemAND.csv', 'problemOR.csv', 'probleXOR.csv']
logic_gate_names = ['AND', 'OR', 'XOR']
logic_gate_paths = [f'C:\\Users\\gigic\\Downloads\\IA-main\\IA-main\\content\\{file}' for file in logic_gate_files]

for gate_name, gate_path in zip(logic_gate_names, logic_gate_paths):
    X_logic, Y_logic = load_logic_gate_data(gate_path)

    # Normalização dos dados das portas lógicas
    X_logic = scaler.fit_transform(X_logic)

    # Configurar validação cruzada
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    fold = 1

    for train_index, val_index in kf.split(X_logic):
        X_train, X_val = X_logic[train_index], X_logic[val_index]
        Y_train, Y_val = Y_logic[train_index], Y_logic[val_index]

        # Treinar o MLP para o problema lógico
        mlp = MLP(input_layer_size=2, hidden_layer_sizes=[2], output_layer_size=1, learning_rate=0.1)
        accuracies, errors, val_accuracies, val_errors = mlp.train(X_train, Y_train, X_val, Y_val, epochs=2000)

        print(f"Fold {fold} - {gate_name} Initial Weights:")
        print(initial_weights)
        print(f"\nFold {fold} - {gate_name} Final Weights:")
        print(final_weights)

        # Plotar as acurácias de treinamento e validação
        plt.figure(figsize=(10, 6))
        plt.plot(accuracies, label=f'Acurácia de Treinamento - Fold {fold} - {gate_name}')
        plt.plot(val_accuracies, label=f'Acurácia de Validação - Fold {fold} - {gate_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.title(f'Acurácia ao longo das Épocas - Fold {fold} - {gate_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"accuracy_plot_{gate_name}_fold_{fold}.png"))
        plt.show()

        # Plotar os erros de treinamento e validação
        plt.figure(figsize=(10, 6))
        plt.plot(errors, label=f'Erro de Treinamento - Fold {fold} - {gate_name}')
        plt.plot(val_errors, label=f'Erro de Validação - Fold {fold} - {gate_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Erro')
        plt.title(f'Erro de Treinamento e Validação ao longo das Épocas - Fold {fold} - {gate_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"error_plot_{gate_name}_fold_{fold}.png"))
        plt.show()

        # Salvar arquivos de saída para cada fold
        np.save(os.path.join(output_dir, f"initial_weights_{gate_name}_fold_{fold}.npy"), initial_weights)
        np.save(os.path.join(output_dir, f"final_weights_{gate_name}_fold_{fold}.npy"), final_weights)
        np.save(os.path.join(output_dir, f"train_errors_{gate_name}_fold_{fold}.npy"), errors)
        np.save(os.path.join(output_dir, f"val_errors_{gate_name}_fold_{fold}.npy"), val_errors)

        fold += 1

# Salvar hiperparâmetros
hyperparameters = {
    "input_layer_size": 120,
    "hidden_layer_sizes": [20, 10],
    "output_layer_size": 26,
    "learning_rate": 0.01,
    "epochs": 2000
}
with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
    for key, value in hyperparameters.items():
        f.write(f"{key}: {value}\n")
