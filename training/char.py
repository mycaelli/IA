import numpy as np
import os
from sklearn.model_selection import KFold

# Funções auxiliares
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Inicializa pesos
        self.weights_input_hidden = np.random.uniform(size=(self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(self.hidden_size, self.output_size))

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output = sigmoid(self.output_layer_input)

        return self.output

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * sigmoid_derivative(output)

        self.hidden_layer_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_layer_delta = self.hidden_layer_error * sigmoid_derivative(self.hidden_layer_output)

        self.weights_hidden_output += self.hidden_layer_output.T.dot(self.output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(self.hidden_layer_delta) * self.learning_rate

    def train(self, X, y, epochs, X_val=None, y_val=None, early_stopping=False, patience=10):
        errors = []
        val_errors = []
        best_val_error = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            error = np.mean(np.square(y - output))
            errors.append(error)

            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_error = np.mean(np.square(y_val - val_output))
                val_errors.append(val_error)

                if early_stopping:
                    if val_error < best_val_error:
                        best_val_error = val_error
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch}")
                            break

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {error}")

        return errors, val_errors

# Função para validação cruzada
def cross_validation(model, X, y, k=5, epochs=1000, early_stopping=False, patience=10):
    fold_size = len(X) // k
    indices = np.random.permutation(len(X))
    X_folds = np.array_split(X[indices], k)
    y_folds = np.array_split(y[indices], k)

    val_errors = []

    for i in range(k):
        X_train = np.concatenate([X_folds[j] for j in range(k) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(k) if j != i])
        X_val = X_folds[i]
        y_val = y_folds[i]

        model_copy = MLP(model.input_size, model.hidden_size, model.output_size, model.learning_rate)
        _, fold_val_errors = model_copy.train(X_train, y_train, epochs, X_val, y_val, early_stopping, patience)
        val_errors.append(fold_val_errors[-1])

    return np.mean(val_errors)

# Carregar dados
data_dir = r'C:\Users\gui02\Downloads\Trab IA\IA-main\logic_gates'
X = np.load(os.path.join(data_dir, 'X.npy'))
Y = np.load(os.path.join(data_dir, 'Y_classe.npy'))

# Verificar e ajustar as dimensões dos dados
if X.ndim == 4:
    X = X.reshape(X.shape[0], -1)
if Y.ndim == 1:
    Y = np.eye(np.max(Y) + 1)[Y]

# Dividir em treinamento, validação e teste
train_size = int(0.6 * len(X))
val_size = int(0.2 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = Y[:train_size], Y[train_size:train_size + val_size], Y[train_size + val_size:]

# Criar e treinar o modelo
mlp = MLP(input_size=X.shape[1], hidden_size=10, output_size=Y.shape[1], learning_rate=0.01)
train_errors, val_errors = mlp.train(X_train, y_train, epochs=1000, X_val=X_val, y_val=y_val, early_stopping=True, patience=10)

# Testar o modelo
test_output = mlp.forward(X_test)
test_error = np.mean(np.square(y_test - test_output))
print(f"Test Error: {test_error}")

# Validação cruzada
cv_error = cross_validation(mlp, X, Y, k=5, epochs=1000, early_stopping=True, patience=10)
print(f"Cross-Validation Error: {cv_error}")

# Salvar arquivos de saída
np.savetxt('pesos_iniciais.txt', mlp.weights_input_hidden)
np.savetxt('pesos_finais.txt', mlp.weights_hidden_output)
np.savetxt('erros_treinamento.txt', train_errors)
np.savetxt('erros_validacao.txt', val_errors)
np.savetxt('saidas_teste.txt', test_output)

# Adicionar lógica para os problemas AND, OR e XOR
def load_logic_gate_data():
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([[0], [0], [0], [1]])  # AND logic gate

    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([[0], [1], [1], [1]])  # OR logic gate

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])  # XOR logic gate

    return (X_and, y_and), (X_or, y_or), (X_xor, y_xor)

# Testar lógica AND, OR e XOR
logic_gates = load_logic_gate_data()
for X_gate, y_gate in logic_gates:
    mlp_logic = MLP(input_size=2, hidden_size=10, output_size=1, learning_rate=0.01)
    mlp_logic.train(X_gate, y_gate, epochs=1000)
    logic_output = mlp_logic.forward(X_gate)
    print(f"Logic Gate Output: {logic_output}")
    np.savetxt(f'saidas_logic_gate_{y_gate[0][0]}.txt', logic_output)
