[02:16, 03/06/2024] Pedro IA: import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

def load_data():
    # Carrega os dados de entrada e saída de arquivos .npy
    input_path = 'X.npy'
    output_path = 'Y_classe.npy'

    input_data = np.load(input_path)
    output_data = np.load(output_path)

    # Ajusta os dados de entrada para uma forma 2D (amostras, características)
    input_data = input_data.reshape(input_data.shape[0], -1)

    return input_data, output_data

def load_logic_gate_data(file_path):
    # Carrega dados de portas lógicas de um arquivo CSV
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    Y = Y.reshape(-1, 1)
    return X, Y

class MLP:
    def _init_(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate=0.01):
        self.hidden_layer_size = hidden_layer_size
        
        # Inicializa os pesos sinápticos com valores aleatórios entre -0.5 e 0.5
        self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, [input_layer_size + 1, hidden_layer_size])
        self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, [hidden_layer_size + 1, output_layer_size])
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        # Função de ativação sigmoide
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivada da função sigmoide para uso na retropropagação
        return x * (1 - x)

    def feedforward(self, X):
        # Propagação direta
        self.input_layer = np.insert(X, 0, 1, axis=1)  # Adiciona bias
        self.hidden_layer_input = np.dot(self.input_layer, self.weights_input_to_hidden)
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.hidden_layer_output = np.insert(self.hidden_layer_output, 0, 1, axis=1)  # Adiciona bias
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_to_output)
        self.output_layer_output = self.sigmoid(self.output_layer_input)
        return self.output_layer_output

    def backpropagation(self, X, Y, output):
        # Retropropagação para ajustar os pesos
        output_error = Y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = np.dot(output_delta, self.weights_hidden_to_output.T[:, 1:])
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output[:, 1:])

        # Atualização dos pesos sinápticos
        self.weights_hidden_to_output += self.learning_rate * np.dot(self.hidden_layer_output.T, output_delta)
        self.weights_input_to_hidden += self.learning_rate * np.dot(self.input_layer.T, hidden_delta)

    def train(self, X_train, Y_train, X_val, Y_val, epochs=1000):
        accuracies = []
        errors = []
        val_accuracies = []
        val_errors = []

        for epoch in range(epochs):
            output = self.feedforward(X_train)
            self.backpropagation(X_train, Y_train, output)

            # Cálculo do erro e acurácia para dados de treinamento
            error = np.mean(np.square(Y_train - output))
            errors.append(error)
            accuracy = accuracy_score(Y_train.argmax(axis=1), np.round(output).argmax(axis=1))
            accuracies.append(accuracy)

            # Cálculo do erro e acurácia para dados de validação
            val_output = self.feedforward(X_val)
            val_error = np.mean(np.square(Y_val - val_output))
            val_errors.append(val_error)
            val_accuracy = accuracy_score(Y_val.argmax(axis=1), np.round(val_output).argmax(axis=1))
            val_accuracies.append(val_accuracy)

        return accuracies, errors, val_accuracies, val_errors

    def predict(self, X):
        # Gera previsões para novos dados
        output = self.feedforward(X)
        return output

# Criar o diretório de saída, se não existir
output_dir = 'output'
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

def train_and_evaluate_model(X, Y, input_layer_size, hidden_layer_size, output_layer_size, learning_rate, epochs, n_splits=5):
    # Normalização dos dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1

    all_accuracies = []
    all_errors = []
    all_val_accuracies = []
    all_val_errors = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]

        mlp = MLP(input_layer_size=input_layer_size, hidden_layer_size=hidden_layer_size, output_layer_size=output_layer_size, learning_rate=learning_rate)
        accuracies, errors, val_accuracies, val_errors = mlp.train(X_train, Y_train, X_val, Y_val, epochs)

        print(f"Fold {fold} - Confusion Matrix:")
        predictions = mlp.predict(X_val)
        cm = confusion_matrix(Y_val.argmax(axis=1), np.round(predictions).argmax(axis=1))
        print(cm)

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

        fold += 1

# Parâmetros da rede neural
input_layer_size = 120
hidden_layer_size = 20
output_layer_size = 26
learning_rate = 0.01
epochs = 2000

# Carregar os dados
X, Y = load_data()

# Treinar e avaliar o modelo
train_and_evaluate_model(X, Y, input_layer_size, hidden_layer_size, output_layer_size, learning_rate, epochs)
[02:21, 03/06/2024] Pedro IA: import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

def load_data():
    input_path = 'X.npy'
    output_path = 'Y_classe.npy'
    input_data = np.load(input_path)
    output_data = np.load(output_path)
    input_data = input_data.reshape(input_data.shape[0], -1)
    return input_data, output_data

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

class MLP:
    def _init_(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate=0.01):
        self.hidden_layer_size = hidden_layer_size
        self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, [input_layer_size + 1, hidden_layer_size])
        self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, [hidden_layer_size + 1, output_layer_size])
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        self.input_layer = np.insert(X, 0, 1, axis=1)
        self.hidden_layer_input = np.dot(self.input_layer, self.weights_input_to_hidden)
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.hidden_layer_output = np.insert(self.hidden_layer_output, 0, 1, axis=1)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_to_output)
        self.output_layer_output = self.sigmoid(self.output_layer_input)
        return self.output_layer_output

    def backpropagation(self, X, Y, output):
        output_error = Y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        hidden_error = np.dot(output_delta, self.weights_hidden_to_output.T[:, 1:])
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output[:, 1:])
        self.weights_hidden_to_output += self.learning_rate * np.dot(self.hidden_layer_output.T, output_delta)
        self.weights_input_to_hidden += self.learning_rate * np.dot(self.input_layer.T, hidden_delta)

    def train(self, X_train, Y_train, epochs=1000):
        accuracies = []
        errors = []
        for epoch in range(epochs):
            output = self.feedforward(X_train)
            self.backpropagation(X_train, Y_train, output)
            error = np.mean(np.square(Y_train - output))
            errors.append(error)
            accuracy = accuracy_score(Y_train.argmax(axis=1), np.round(output).argmax(axis=1))
            accuracies.append(accuracy)
        return accuracies, errors

    def predict(self, X):
        output = self.feedforward(X)
        return output

def early_stopping(accuracies, threshold=0.01, patience=10):
    if len(accuracies) < patience + 1:
        return False
    for i in range(patience):
        if accuracies[-1 - i] - accuracies[-patience - 1] > threshold:
            return False
    return True

def train_with_cross_validation(X, Y, hidden_layer_size=20, learning_rate=0.01, epochs=2000, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_errors = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]

        mlp = MLP(input_layer_size=X_train.shape[1], hidden_layer_size=hidden_layer_size, output_layer_size=Y_train.shape[1], learning_rate=learning_rate)
        val_accuracies = []
        val_errors = []

        for epoch in range(epochs):
            mlp.train(X_train, Y_train, epochs=1)
            val_output = mlp.predict(X_val)
            val_accuracy = accuracy_score(Y_val.argmax(axis=1), np.round(val_output).argmax(axis=1))
            val_error = np.mean(np.square(Y_val - val_output))
            val_accuracies.append(val_accuracy)
            val_errors.append(val_error)

            if early_stopping(val_accuracies):
                print(f"Parada antecipada no epoch {epoch}")
                break

        fold_accuracies.append(val_accuracies)
        fold_errors.append(val_errors)

    return fold_accuracies, fold_errors

# Carregar dados
X, Y = load_data()

# Separar os últimos 130 exemplos para teste
X_train, X_test = X[:-130], X[-130:]
Y_train, Y_test = Y[:-130], Y[-130:]

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o modelo com validação cruzada e parada antecipada
fold_accuracies, fold_errors = train_with_cross_validation(X_train, Y_train, hidden_layer_size=20, learning_rate=0.01, epochs=2000, n_splits=5)

# Plotar resultados
plt.figure(figsize=(10, 5))
for i in range(len(fold_accuracies)):
    plt.plot(fold_accuracies[i], label=f'Fold {i+1}')
plt.title('Acurácia ao longo das épocas (Validação Cruzada)')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for i in range(len(fold_errors)):
    plt.plot(fold_errors[i], label=f'Fold {i+1}')
plt.title('Erro ao longo das épocas (Validação Cruzada)')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.legend()
plt.show()

# Avaliar o modelo nos dados de teste
mlp = MLP(input_layer_size=X_train.shape[1], hidden_layer_size=20, output_layer_size=Y_train.shape[1], learning_rate=0.01)
mlp.train(X_train, Y_train, epochs=2000)
y_pred = np.round(mlp.predict(X_test)).argmax(axis=1)
plot_confusion_matrix(Y_test.argmax(axis=1), y_pred, classes=[chr(i) for i in range(65, 91)], title='Confusion Matrix - Com Validação Cruzada')