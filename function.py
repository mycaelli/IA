import math
import numpy as np

# Função de ativação sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Calcula a derivada da função sigmoid."""
    return sigmoid(x) * (1 - sigmoid(x))

def safe_sigmoid(x):
    """Calcula a função sigmoid de forma segura para um escalar ou elementos de uma lista."""
    # Verifica se 'x' é uma lista e aplica a função sigmoid a cada elemento
    if isinstance(x, list):
        return [safe_sigmoid(item) for item in x]
    else:
        # Aqui, garantimos que 'x' é um escalar antes de passá-lo para math.exp
        return 1 / (1 + math.exp(-x))