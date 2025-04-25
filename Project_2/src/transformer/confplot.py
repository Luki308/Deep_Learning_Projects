import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the confusion matrix from the pickle file


def main():
    with open('23.04-20.05matrix.pickle', 'rb') as file:
        confusion_matrix, label_mapping = pickle.load(file)

    # Convert numerical labels to text labels
    text_labels = list(label_mapping.keys())

    # Plot the confusion matrix with text labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', square=True,
                xticklabels=text_labels, yticklabels=text_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Plot the confusion matrix
if __name__ == '__main__':
    main()