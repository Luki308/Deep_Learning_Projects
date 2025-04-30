import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def run(model, scenario):
    folder_path = rf"D:\Msc\sem1\DL\AUDIO\Deep_Learning_Projects\Project_2\src\results\{model}\{scenario}"

    pickle_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith("matrix.pickle")]

    if not pickle_files:
        print(f"No confusion matrices in {folder_path}")
        exit()

    confusion_matrices = []
    label_mapping = None

    for file in pickle_files:
        try:
            with open(file, "rb") as f:
                matrix, mapping = pickle.load(f)
                confusion_matrices.append(matrix)
                if label_mapping is None:
                    label_mapping = mapping  # use first mapping
                else:
                    assert label_mapping == mapping
        except Exception: # one case
            with open(file, "rb") as f:
                matrix = pickle.load(f)
                confusion_matrices.append(matrix)

    text_labels = list(label_mapping.keys())

    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
    percentage_confusion_matrix = mean_confusion_matrix / mean_confusion_matrix.sum(axis=0, keepdims=True) * 100

    # plot the mean
    plt.figure(figsize=(8, 6))
    sns.heatmap(mean_confusion_matrix, annot=True, cmap="Blues", square=True,
                xticklabels=text_labels, yticklabels=text_labels)
    plt.title(f"{model}-{scenario}: Mean Confusion Matrix Across Runs")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # save
    plot_path = os.path.join(folder_path, "mean_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()

    # plot the %
    plt.figure(figsize=(8, 6))
    sns.heatmap(percentage_confusion_matrix, annot=True, cmap="Blues", square=True,
                xticklabels=text_labels, yticklabels=text_labels)
    plt.title(f"{model}-{scenario}: Mean Percentage Prediction Confusion Matrix Across Runs")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Save the plot
    plot_path = os.path.join(folder_path, "mean_pr_confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")


def main():
    experiments = {
        "CT": ["h4l2", "h8l2", "h16l2", "h16l3", "h16l1", "h8l2_cl11"],
        "TSTF": ["h2l2", "h4l2", "h8l2", "h16l2", "h16l3"]
    }
    for model in experiments.keys():
        for scenario in experiments[model]:
            run(model, scenario)


if __name__ == '__main__':
    main()
