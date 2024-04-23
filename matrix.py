import numpy as np

# Operações básicas de matriz
def initialize_matrix(rows, cols):
    """Inicializa uma matriz com valores aleatórios entre -1 e 1."""
    return np.random.uniform(-1, 1, (rows, cols))

def array_product(array_a, array_b):
    """Calcula o produto escalar de dois vetores."""
    #  número de colunas da primeira matriz seja igual ao número de linhas da segunda matriz
    return np.dot(array_a, array_b)


# Função para somar elementos de duas listas elemento a elemento
def elementwise_sum(matrix_a, matrix_b):
    """Soma dois vetores ou matrizes elemento a elemento."""
    return matrix_a + matrix_b

# matrix.reshape(true_output, predicted_output)
def reshape(matrix_a):
    """Redimensiona a matrix_b para ter o mesmo número de linhas que a matrix_a"""
    matrix_a.reshape(-1, 1)
    return matrix_a


# Função para somar elementos de duas listas elemento a elemento
def elementwise_subtraction(matrix_a, matrix_b):
    """Soma dois vetores ou matrizes elemento a elemento."""
    return matrix_a - matrix_b

def multiply(matrix_a, matrix_b):
    """Multiplica duas matrizes."""
    return np.matmul(matrix_a, matrix_b)

def sum(matrix, axis=None, keepdims=False):
    """Soma todos os elementos de uma matrix (ou de um eixo)"""
    return np.sum(matrix, axis=axis, keepdims=keepdims)

def apply_function(func, matrix):
    """Aplica uma função a cada elemento de uma matriz."""
    return np.vectorize(func)(matrix)

def transpose(matrix):
    """Transpõe uma matriz."""
    return np.transpose(matrix)

def scalar_multiply(scalar, matrix):
    """Multiplica cada elemento de uma matriz por um escalar."""
    # Cada elemento da matriz (ou seja, cada sublista) é multiplicado pelo escalar
    return scalar * matrix

def mean(matrix_a, matrix_b):
    return np.mean((matrix_a - matrix_b) ** 2)
