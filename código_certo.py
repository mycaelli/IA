import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

def load_data():
    """
    Função para carregar dados de entrada e saída a partir de arquivos .npy.
    """
    input_path = 'C:\\Users\\gui02\\Downloads\\EP IA\\IA-main\\content\\X.npy'  # Caminho para o arquivo de dados de entrada
    output_path = 'C:\\Users\\gui02\\Downloads\\EP IA\\IA-main\\content\\Y_classe.npy'  # Caminho para o arquivo de dados de saída

    input_data = np.load(input_path)  # Carrega dados de entrada
    output_data = np.load(output_path)  # Carrega dados de saída

    # Ajusta os dados de entrada para uma forma 2D (n_amostras, n_características)
    input_data = input_data.reshape(input_data.shape[0], -1)

    return input_data, output_data

class MLP:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate=0.01):
        """
        Inicializa uma Rede Neural Perceptron Multicamadas (MLP).
        """
        self.hidden_layer_size = hidden_layer_size  # Define o tamanho da camada escondida

        # Inicializa pesos sinápticos com valores aleatórios entre -0.5 e 0.5
        self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, [input_layer_size + 1, hidden_layer_size])
        self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, [hidden_layer_size + 1, output_layer_size])
        self.learning_rate = learning_rate  # Define a taxa de aprendizado

    def sigmoid(self, x):
        """
        Função de ativação sigmoide.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivada da função sigmoide.
        """
        return x * (1 - x)

    def feedforward(self, X):
        """
        Realiza a propagação direta dos dados de entrada através da rede.
        """
        self.input_layer = np.insert(X, 0, 1, axis=1)  # Adiciona o bias aos dados de entrada
        self.hidden_layer_input = np.dot(self.input_layer, self.weights_input_to_hidden)
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.hidden_layer_output = np.insert(self.hidden_layer_output, 0, 1, axis=1)  # Adiciona o bias à camada escondida
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_to_output)
        self.output_layer_output = self.sigmoid(self.output_layer_input)
        return self.output_layer_output

    def backpropagation(self, X, Y, output):
        """
        Realiza o processo de retropropagação para ajustar os pesos sinápticos.
        """
        output_error = Y - output  # Calcula o erro na camada de saída
        output_delta = output_error * self.sigmoid_derivative(output)  # Calcula o delta da saída

        hidden_error = np.dot(output_delta, self.weights_hidden_to_output.T[:, 1:])  # Calcula o erro na camada escondida
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output[:, 1:])  # Calcula o delta da camada escondida

        # Atualiza os pesos sinápticos
        self.weights_hidden_to_output += self.learning_rate * np.dot(self.hidden_layer_output.T, output_delta)
        self.weights_input_to_hidden += self.learning_rate * np.dot(self.input_layer.T, hidden_delta)

    def train(self, X_train, Y_train, X_val, Y_val, epochs=1000):
        """
        Treina a rede neural usando os dados de treinamento e validação.
        """
        accuracies = []
        errors = []
        val_accuracies = []
        val_errors = []

        for epoch in range(epochs):
            output = self.feedforward(X_train)  # Realiza a propagação direta
            self.backpropagation(X_train, Y_train, output)  # Realiza a retropropagação

            # Calcula o erro e a acurácia dos dados de treinamento
            error = np.mean(np.square(Y_train - output))
            errors.append(error)
            accuracy = accuracy_score(Y_train.argmax(axis=1), np.round(output).argmax(axis=1))
            accuracies.append(accuracy)

            # Validação
            val_output = self.feedforward(X_val)  # Propagação direta nos dados de validação
            val_error = np.mean(np.square(Y_val - val_output))  # Calcula o erro de validação
            val_errors.append(val_error)
            val_accuracy = accuracy_score(Y_val.argmax(axis=1), np.round(val_output).argmax(axis=1))  # Calcula a acurácia de validação
            val_accuracies.append(val_accuracy)

        return accuracies, errors, val_accuracies, val_errors

    def predict(self, X):
        """
        Gera previsões para novos dados.
        """
        output = self.feedforward(X)
        return output

# Criar o diretório de saída, se não existir
output_dir = 'C:\\Users\\gui02\\Downloads\\EP IA\\IA-main\\arquivos de saida'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carregar os dados dos caracteres completos
X, Y = load_data()

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Manter os últimos 130 valores para teste
X_test = X[-130:]
Y_test = Y[-130:]
X_remaining = X[:-130]
Y_remaining = Y[:-130]

# Dividir os dados restantes em treino (60%) e validação (40%)
X_train, X_val, Y_train, Y_val = train_test_split(X_remaining, Y_remaining, test_size=0.4, random_state=42)

# Configurar validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

# Listas para armazenar os resultados de cada fold
all_accuracies = []
all_errors = []
all_val_accuracies = []
all_val_errors = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Divide os dados de treinamento em treinamento e validação
    Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]

    # Treinar o MLP com ajustes
    mlp = MLP(input_layer_size=120, hidden_layer_size=20, output_layer_size=26, learning_rate=0.01)
    initial_weights = {'weights_input_to_hidden': mlp.weights_input_to_hidden.copy(), 'weights_hidden_to_output': mlp.weights_hidden_to_output.copy()}
    accuracies, errors, val_accuracies, val_errors = mlp.train(X_train_fold, Y_train_fold, X_val_fold, Y_val_fold, epochs=2000)
    final_weights = {'weights_input_to_hidden': mlp.weights_input_to_hidden, 'weights_hidden_to_output': mlp.weights_hidden_to_output}

    # Exibir pesos iniciais e finais
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

# Treinar o MLP com os dados completos de treinamento e validação
mlp_final = MLP(input_layer_size=120, hidden_layer_size=20, output_layer_size=26, learning_rate=0.01)
mlp_final.train(X_train, Y_train, X_val, Y_val, epochs=2000)

# Fazer previsões no conjunto de teste
test_predictions = mlp_final.predict(X_test)

# Calcular a acurácia no conjunto de teste
test_accuracy = accuracy_score(Y_test.argmax(axis=1), np.round(test_predictions).argmax(axis=1))
print(f"Acurácia no conjunto de teste: {test_accuracy}")

# Salvar hiperparâmetros
hyperparameters = {
    "input_layer_size": 120,
    "hidden_layer_size": 20,
    "output_layer_size": 26,
    "learning_rate": 0.01,
    "epochs": 2000
}
with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
    for key, value in hyperparameters.items():
        f.write(f"{key}: {value}\n")
