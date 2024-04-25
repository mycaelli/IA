import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_loss(loss_values, epoch, learning_rate):
    os.makedirs(f'plots/learning_rate_{learning_rate}', exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(loss_values)+1), loss_values, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(f'plots/learning_rate_{learning_rate}/epoch_{epoch}_loss.png')
    plt.close()


def plot_metrics(accuracy_values, precision_values, recall_values, epoch, learning_rate):
    os.makedirs(f'plots/learning_rate_{learning_rate}', exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(accuracy_values)+1), accuracy_values, label='Accuracy')
    plt.plot(range(1, len(precision_values)+1), precision_values, label='Precision')
    plt.plot(range(1, len(recall_values)+1), recall_values, label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Training Metrics')
    plt.legend()
    plt.savefig(f'plots/learning_rate_{learning_rate}/epoch_{epoch}_metrics.png')
    plt.close()

def plot_confusion_matrix(output_prediction, expected_output, epoch, learning_rate):
    num_classes = 2
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

    # Preenche a matriz de confusão
    for pred, true in zip(output_prediction, expected_output):
        pred = int(pred)  # Converte para inteiro
        true = int(true)  # Converte para inteiro
        confusion_mat[true, pred] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.xticks(np.arange(num_classes), labels=np.arange(num_classes))
    plt.yticks(np.arange(num_classes), labels=np.arange(num_classes))
    plt.savefig(f'plots/learning_rate_{learning_rate}/epoch_{epoch}_confusion_matrix.png')
    plt.close()
