import matplotlib.pyplot as plt
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
