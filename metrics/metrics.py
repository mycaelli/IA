import ia.matrix as matrix 

# tem que adicionar informacoes pra conseguir reproduzir o resultado depois

def calculate_metrics(predicted_output, expected_output):
  loss = matrix.mean(predicted_output, expected_output)
  predicted_classes = (predicted_output > 0.5).astype(int)
  true_positives = matrix.sum((predicted_classes == 1) & (expected_output == 1))
  false_positives = matrix.sum((predicted_classes == 1) & (expected_output == 0))
  true_negatives = matrix.sum((predicted_classes == 0) & (expected_output == 0))
  false_negatives = matrix.sum((predicted_classes == 0) & (expected_output == 1)) 
  accuracy = (true_positives + true_negatives) / len(expected_output)
  precision = true_positives / (true_positives + false_positives + 1e-9)
  recall = true_positives / (true_positives + false_negatives + 1e-9) 
  
  return loss, accuracy, precision, recall