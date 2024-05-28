import ia.matrix as matrix
import numpy as np
# tem que adicionar informacoes pra conseguir reproduzir o resultado depois

def calculate_metrics(output_prediction, expected_output):
  TP, FP, TN, FN = 0, 0, 0, 0

  for pred, exp in zip(output_prediction, expected_output):
    if pred == 1 and exp == 1:
        TP += 1
    elif pred == 1 and exp == -1:
        FP += 1
    elif pred == -1 and exp == -1:
        TN += 1
    elif pred == -1 and exp == 1:
        FN += 1

  P = TP + FP
  N = TN + FN
  
  error = 0 # FP + FN / (P + N)
        
  accuracy = 0 # TP + TN / (P + N)

  precision = TP/(TP + FP)

  recall = TP / (TP + FN)

  return error, accuracy, precision, recall
