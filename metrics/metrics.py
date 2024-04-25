import ia.matrix as matrix 
import numpy as np
# tem que adicionar informacoes pra conseguir reproduzir o resultado depois

def calculate_metrics(output_prediction, expected_output):

  loss = loss = matrix.sum(output_prediction**2)
  # print('Predction, expected', output_prediction, expected_output)
  true_positives = np.sum((output_prediction == 1) & (expected_output == 1))
  false_positives = np.sum((output_prediction == 1) & (expected_output == 0))
  true_negatives = np.sum((output_prediction == 0) & (expected_output == 0))
  false_negatives = np.sum((output_prediction == 0) & (expected_output == 1))

  # print(true_positives)
  # print(false_positives)
  # print(true_negatives)
  # print(false_negatives)

  accuracy = (true_positives + true_negatives) / len(expected_output) # acuracia

  if true_positives + false_positives != 0:
    precision = true_positives / (true_positives + false_positives) # precisao
  else:
    precision = 0.0

  if true_positives + false_negatives != 0:
    recall = true_positives / (true_positives + false_negatives) # sensibilidade
  else:
    recall = 0.0
  # specificity = true_negatives / (false_positives + true_negatives) # especificidade
  
  return loss, accuracy, precision, recall
