import csv
import ia.matrix as matrix 

def read_csv(file_path):
    data = []
    with open(file_path, "r", encoding='utf-8-sig') as file:
        file_csv = csv.reader(file)
        for line in file_csv:
            data.append(line)

    return convert_input_to_int(data)
    
def convert_input_to_int(data):
    # converte os dados pra inteiro
    converted_data = [[int(v) for v in line] for line in data] 
    return converted_data


def logic_gate_problems(int_data):
    data_input = matrix.np.array([line[:-1] for line in int_data])
    expected_output = matrix.np.array([line[-1] for line in int_data]).reshape(-1, 1)
    return  data_input, expected_output
