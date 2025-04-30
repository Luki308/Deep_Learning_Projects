import pandas as pd
import matplotlib.pyplot as plt
import os

def run(model, scenario):
    csv_path = rf"D:\Msc\sem1\DL\AUDIO\Deep_Learning_Projects\Project_2\src\results\{model}\{scenario}\aggregated_results.csv"

    df = pd.read_csv(csv_path)

    accuracy_metrics = ["Accuracy_train", "Accuracy_validation"]
    loss_metrics = ["Loss_train", "Loss_validation"]

    def plot_metrics(metric_list, title, filename, metric_name):
        plt.figure(figsize=(10, 5))

        for metric in metric_list:
            subset = df[df["Metric"] == metric]
            if not subset.empty:
                plt.plot(subset["Epoch"], subset["Mean"], label=f"{metric} (Mean)")
                plt.fill_between(subset["Epoch"], subset["Mean"] - subset["Std Dev"],
                                 subset["Mean"] + subset["Std Dev"], alpha=0.3)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel("Epoch")
        plt.ylabel(f"{metric_name} Value")
        plt.title(title)
        plt.legend()
        plt.grid(True)

        # save
        plot_path = os.path.join(os.path.dirname(csv_path), filename)
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to: {plot_path}")
        plt.close()

    # accuracy plot
    plot_metrics(accuracy_metrics, f"{model}-{scenario}: Training vs Validation Accuracy", "accuracy_plot.png", "Accuracy")

    # loss plot
    plot_metrics(loss_metrics, f"{model}-{scenario}: Training vs Validation Loss", "loss_plot.png", "Loss")


def main():
    experiments ={
        "CT": ["h4l2", "h8l2", "h16l2", "h16l3", "h16l1"],
        "TSTF": ["h2l2", "h4l2", "h8l2", "h16l2", "h16l3", "h16l1"]
    }
    for model in experiments.keys():
        for scenario in experiments[model]:
            run(model, scenario)


if __name__ == '__main__':
    main()