
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data():
  '''
  Função que carrega os dados para validação dos caracteres
  Garante que os dados estejam em uma única dimensão para que possam ser usados no treinamento
  '''
  input = np.load('./content/X.npy') # dados vem no formato de matrix
  output_data = np.load('./content/Y_classe.npy')
  input_data = []
  for d in input:
    character = d.flatten()
    input_data.append(character)
  return input_data, output_data

X, Y = load_data()
X = np.array(X)

input_layer_size = 120
hidden_layer_size = 3
output_layer_size = Y.shape[1] # 26 letras

class MLP:
  def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate=0.01):
    '''
    Inicialização dos pesos (aleatória e uniformizada), bias e learning rate da rede
    '''
    self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, [input_layer_size, hidden_layer_size])
    self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, [hidden_layer_size, output_layer_size])
    self.bias = np.full(hidden_layer_size, 1, dtype=float)
    self.learning_rate = learning_rate

  def sigmoid(self, x):
    'função sigmoide que é utilizada como função de ativação'
    return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(self, x):
    'derivada da função de ativação'
    return x * (1 - x)
  
  def feedforward(self, X):
    'função que proporciona o resultado do treinamento da rede'
    weighted_sum_input_to_hidden = np.dot(X, self.weights_input_to_hidden) + self.bias # soma ponderada
    self.hidden_output = self.sigmoid(weighted_sum_input_to_hidden)
    weighted_sum_hidden_to_output = np.dot(self.hidden_output, self.weights_hidden_to_output) # soma ponderada
    output = self.sigmoid(weighted_sum_hidden_to_output)
    return output

  def backpropagation(self, X, Y, output):
    'função que atualiza os pesos da rede de acordo com os erros encontrados nos resultados obtidos pelo feedfoward'
    output_error = Y - output
    output_delta = output_error * self.sigmoid_derivative(output)
    hidden_error = np.dot(output_delta, self.weights_hidden_to_output.T)
    hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
    self.weights_hidden_to_output += np.dot(self.hidden_output.T, output_delta) * self.learning_rate
    self.weights_input_to_hidden += np.dot(X.T, hidden_delta) * self.learning_rate
    self.bias += np.sum(hidden_delta, axis=0) * self.learning_rate

  def train(self, X, Y, epochs=1000):
    accuracies = []
    errors = []

    for epoch in range(epochs):
      output = self.feedforward(X)
      self.backpropagation(X, Y, output)

      error = np.mean(np.square(Y - output))
      errors.append(error)

      accuracy = accuracy_score(Y.argmax(axis=1), np.round(output).argmax(axis=1))
      accuracies.append(accuracy)
      # print(f'Epoch {epoch}, errors {error} accuracy {accuracy}')

    return accuracies, errors

mlp = MLP(120, 3, 26, 0.04)
initial_weights = {'weights_input_to_hidden': mlp.weights_input_to_hidden, 'bias': mlp.bias, 'weights_hidden_to_output': mlp.weights_hidden_to_output}
accuracies, errors  = mlp.train(X, Y, 1000)
final_weights = {'weights_input_to_hidden': mlp.weights_input_to_hidden, 'bias': mlp.bias, 'weights_hidden_to_output': mlp.weights_hidden_to_output}

print(initial_weights)
print()
print(final_weights)

# Plotar as acurácias de treinamento e validação
plt.figure(figsize=(10, 6))
plt.plot(accuracies, label='Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.title('Acurácia ao longo das Épocas')
plt.legend()
plt.show()

# Plotar os erros de treinamento
plt.figure(figsize=(10, 6))
plt.plot(errors, label='Erro de Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.title('Erro de Treinamento ao longo das Épocas')
plt.legend()
plt.show()

