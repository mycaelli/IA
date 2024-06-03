import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def load_data():
    input_path = 'C:\\Users\\gui02\\Downloads\\EP IA\\IA-main\\content\\X.npy'
    output_path = 'C:\\Users\\gui02\\Downloads\\EP IA\\IA-main\\content\\Y_classe.npy'

    input_data = np.load(input_path)
    output_data = np.load(output_path)

    input_data = input_data.reshape(input_data.shape[0], -1)

    return input_data, output_data

class MLP:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate=0.01):
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

    def train(self, X_train, Y_train, X_val, Y_val, epochs=1000):
        accuracies = []
        errors = []
        val_accuracies = []
        val_errors = []

        best_val_accuracy = 0
        epochs_no_improve = 0

        for epoch in range(epochs):
            output = self.feedforward(X_train)
            self.backpropagation(X_train, Y_train, output)

            error = np.mean(np.square(Y_train - output))
            errors.append(error)
            accuracy = self.calculate_accuracy(Y_train, output)
            accuracies.append(accuracy)

            val_output = self.feedforward(X_val)
            val_error = np.mean(np.square(Y_val - val_output))
            val_errors.append(val_error)
            val_accuracy = self.calculate_accuracy(Y_val, val_output)
            val_accuracies.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        return accuracies, errors, val_accuracies, val_errors

    def predict(self, X):
        output = self.feedforward(X)
        return output

    def calculate_accuracy(self, Y_true, Y_pred):
        correct_predictions = np.sum(Y_true.argmax(axis=1) == Y_pred.argmax(axis=1))
        accuracy = correct_predictions / Y_true.shape[0]
        return accuracy

    def plot_metrics(self, accuracies, errors, val_accuracies, val_errors, fold, output_dir):
        # Plotar as acurácias de treinamento e validação
        plt.figure(figsize=(10, 6))
        plt.plot(accuracies, label=f'Training Accuracy - Fold {fold}')
        plt.plot(val_accuracies, label=f'Validation Accuracy - Fold {fold}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy over Epochs - Fold {fold}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"accuracy_plot_fold_{fold}.png"))
        plt.show()

        # Plotar os erros de treinamento e validação
        plt.figure(figsize=(10, 6))
        plt.plot(errors, label=f'Training Error - Fold {fold}')
        plt.plot(val_errors, label=f'Validation Error - Fold {fold}')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(f'Training and Validation Error over Epochs - Fold {fold}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"error_plot_fold_{fold}.png"))
        plt.show()

    def plot_confusion_matrix(self, Y_true, Y_pred, fold, output_dir):
        cm = np.zeros((Y_true.shape[1], Y_true.shape[1]), dtype=int)
        for true, pred in zip(Y_true.argmax(axis=1), Y_pred.argmax(axis=1)):
            cm[true, pred] += 1

        plt.figure(figsize=(10, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.colorbar()
        tick_marks = np.arange(Y_true.shape[1])
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_fold_{fold}.png"))
        plt.show()

def grid_search(X, Y, param_grid):
    best_params = None
    best_accuracy = 0
    results = []

    num_folds = 5
    fold_size = len(X) // num_folds

    for learning_rate in param_grid['learning_rate']:
        for hidden_layer_size in param_grid['hidden_layer_sizes']:
            fold_accuracies = []
            for fold in range(num_folds):
                start = fold * fold_size
                end = (fold + 1) * fold_size if fold != num_folds - 1 else len(X)

                X_val_fold = X[start:end]
                Y_val_fold = Y[start:end]
                X_train_fold = np.concatenate([X[:start], X[end:]])
                Y_train_fold = np.concatenate([Y[:start], Y[end:]])

                mlp = MLP(input_layer_size=X_train_fold.shape[1], hidden_layer_size=hidden_layer_size,
                          output_layer_size=Y_train_fold.shape[1], learning_rate=learning_rate)
                _, _, val_accuracies, _ = mlp.train(X_train_fold, Y_train_fold, X_val_fold, Y_val_fold, epochs=2000)
                fold_accuracies.append(np.mean(val_accuracies))

            mean_val_accuracy = np.mean(fold_accuracies)
            results.append({
                'learning_rate': learning_rate,
                'hidden_layer_size': hidden_layer_size,
                'val_accuracy': mean_val_accuracy
            })

            if mean_val_accuracy > best_accuracy:
                best_accuracy = mean_val_accuracy
                best_params = {'learning_rate': learning_rate, 'hidden_layer_size': hidden_layer_size}

    return best_params, results

# Criar o diretório de saída, se não existir
output_dir = 'C:\\Users\\gui02\\Downloads\\EP IA\\IA-main\\arquivos de saida'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carregar os dados
X, Y = load_data()

# Normalização dos dados
scaler = lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0)
X = scaler(X)

# Manter os últimos 130 valores para teste
X_test = X[-130:]
Y_test = Y[-130:]
X_remaining = X[:-130]
Y_remaining = Y[:-130]

# Dividir os dados restantes em treino (60%) e validação (40%) manualmente
def train_test_split_manual(X, Y, test_size):
    n_train = int((1 - test_size) * X.shape[0])
    indices = np.random.permutation(X.shape[0])
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    return X[train_indices], X[val_indices], Y[train_indices], Y[val_indices]

X_train, X_val, Y_train, Y_val = train_test_split_manual(X_remaining, Y_remaining, test_size=0.4)

# Definir a grade de parâmetros para o Grid Search
param_grid = {
    'learning_rate': [0.001, 0.005, 0.01],
    'hidden_layer_sizes': [20, 40, 60]
}

# Executar o Grid Search
best_params, results = grid_search(X_train, Y_train, param_grid)

# Exibir os melhores parâmetros
print("Melhores parâmetros encontrados:", best_params)

# Treinar o MLP com os melhores parâmetros encontrados
mlp = MLP(input_layer_size=120, hidden_layer_size=best_params['hidden_layer_size'], output_layer_size=26,
          learning_rate=best_params['learning_rate'])
initial_weights = {'weights_input_to_hidden': mlp.weights_input_to_hidden.copy(), 'weights_hidden_to_output': mlp.weights_hidden_to_output.copy()}
accuracies, errors, val_accuracies, val_errors = mlp.train(X_train, Y_train, X_val, Y_val, epochs=2000)
final_weights = {'weights_input_to_hidden': mlp.weights_input_to_hidden, 'weights_hidden_to_output': mlp.weights_hidden_to_output}

# Exibir pesos iniciais e finais
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
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
plt.show()

# Plotar os erros de treinamento e validação
plt.figure(figsize=(10, 6))
plt.plot(errors, label='Erro de Treinamento')
plt.plot(val_errors, label='Erro de Validação')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.title('Erro de Treinamento e Validação ao longo das Épocas')
plt.legend()
plt.savefig(os.path.join(output_dir, "error_plot.png"))
plt.show()

# Salvar arquivos de saída
np.save(os.path.join(output_dir, "initial_weights.npy"), initial_weights)
np.save(os.path.join(output_dir, "final_weights.npy"), final_weights)
np.save(os.path.join(output_dir, "train_errors.npy"), errors)
np.save(os.path.join(output_dir, "val_errors.npy"), val_errors)

# Treinar o MLP com os dados completos de treinamento e validação
mlp_final = MLP(input_layer_size=120, hidden_layer_size=best_params['hidden_layer_size'], output_layer_size=26, learning_rate=best_params['learning_rate'])
mlp_final.train(X_train, Y_train, X_val, Y_val, epochs=2000)

# Fazer previsões no conjunto de teste
test_predictions = mlp_final.predict(X_test)

# Calcular a acurácia no conjunto de teste
test_accuracy = mlp_final.calculate_accuracy(Y_test, test_predictions)
print(f"Acurácia no conjunto de teste: {test_accuracy}")

# Plotar a matriz de confusão para o conjunto de teste
mlp_final.plot_confusion_matrix(Y_test, test_predictions, 'Test', output_dir)

# Salvar hiperparâmetros
hyperparameters = {
    "input_layer_size": 120,
    "hidden_layer_size": best_params['hidden_layer_size'],
    "output_layer_size": 26,
    "learning_rate": best_params['learning_rate'],
    "epochs": 2000
}
with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as f:
    for key, value in hyperparameters.items():
        f.write(f"{key}: {value}\n")