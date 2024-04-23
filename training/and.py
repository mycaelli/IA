import sys
sys.path.append('../')  # Adiciona o diretório base ao Python Path

from ia.input import *
import ia.input as input
import ia.mlp as mlp

FILE_PATH = "./logic_gates/problemAND.csv"

# X input
# y expected output
data = input.read_csv(FILE_PATH)
X, y = input.logic_gate_problems(data)

# input_size = numero de neuronios de entrada (ex: numeros de caracteres 1, -1)
# hidden_size = numero de neuronios na camada oculta
# output_size = ex: problema binario tem uma saida, problemas multiclasse podem ter muitas
model = mlp.MLP(input_size=2, hidden_size=3, output_size=1, learning_rate=0.05)
model.train(input_data=X, expected_output=y, epochs=100)