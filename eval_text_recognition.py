import numpy as np
import Levenshtein
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def calculate_cer(targets, predictions):
    # Calculate CER score
    total_chars = 0
    total_distance = 0
    for target, pred in zip(targets, predictions):
        distance = Levenshtein.distance(target, pred)
        total_distance += distance # Number of incorrectly predicted characters
        total_chars += len(target)

    cer = total_distance / total_chars
    return cer

def calculate_wer(targets, predictions):
    # Calculate WER score
    incorrect_words = sum([1 if target != pred else 0 for target, pred in zip(targets, predictions)])    
    wer = incorrect_words / len(targets)
    return wer

def get_target_prediction(path):
    targets = []
    predictions = []
    with open(path, 'r', encoding='utf-8') as text:
        lines = text.readlines()
        for line in lines:
            target = line.split(",")[1].strip()
            prediction = line.split(",")[2].strip()
            targets.append(target)
            predictions.append(prediction)
    return targets, predictions


if __name__ == "__main__":
    path = r"...\AttentionHTR\eval.txt"
    targets, predictions = get_target_prediction(path)

    cer = calculate_cer(targets, predictions)
    wer = calculate_wer(targets, predictions)
 
    print(f"CER (Character Error Rate): {cer:.4f}")
    print(f"WER (Word Error Rate): {wer:.4f}")

    # Create confusion matrix for recogniced MNIST digits
    labels = sorted(set(targets))

    conf_matrix = confusion_matrix(targets, predictions, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("MNIST Prediction Confusion Matrix")
    plt.tight_layout() 

    plot_filename = r"...\MNIST_pix2pix_conf.png"
    plt.savefig(plot_filename)
    plt.show()