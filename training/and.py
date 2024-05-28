import numpy as np
import pandas as pd

# Funções de portas lógicas
def logic_and(a, b):
    return int(a and b)

def logic_or(a, b):
    return int(bool(a) or bool(b))

def logic_xor(a, b):
    return 1 if a != b else -1

# Caminhos para os arquivos CSV
problem_and_path = r'C:\Users\gui02\Downloads\Trab IA\IA-main\logic_gates\problemAND.csv'
problem_or_path = r'C:\Users\gui02\Downloads\Trab IA\IA-main\logic_gates\problemOR.csv'
problem_xor_path = r'C:\Users\gui02\Downloads\Trab IA\IA-main\logic_gates\problemXOR.csv'

# Função para carregar dados e aplicar função lógica
def apply_logic_gate(filepath, logic_function):
    df = pd.read_csv(filepath, header=None)
    df.columns = ['Input1', 'Input2', 'Expected']
    df['Predicted'] = df.apply(lambda row: logic_function(row['Input1'], row['Input2']), axis=1)
    df['Correct'] = df['Predicted'] == df['Expected']
    accuracy = df['Correct'].mean()
    return df, accuracy

# Aplicar funções lógicas aos datasets
df_and, accuracy_and = apply_logic_gate(problem_and_path, logic_and)
df_or, accuracy_or = apply_logic_gate(problem_or_path, logic_or)
df_xor, accuracy_xor = apply_logic_gate(problem_xor_path, logic_xor)

# Salvar resultados em arquivos de saída
df_and.to_csv('resultados_and.csv', index=False)
df_or.to_csv('resultados_or.csv', index=False)
df_xor.to_csv('resultados_xor.csv', index=False)

# Exibir acurácias
print(f"Accuracy for AND gate: {accuracy_and}")
print(f"Accuracy for OR gate: {accuracy_or}")
print(f"Accuracy for XOR gate: {accuracy_xor}")

# Salvar acurácias em um arquivo
with open('acuracias.txt', 'w') as f:
    f.write(f"Accuracy for AND gate: {accuracy_and}\n")
    f.write(f"Accuracy for OR gate: {accuracy_or}\n")
    f.write(f"Accuracy for XOR gate: {accuracy_xor}\n")
