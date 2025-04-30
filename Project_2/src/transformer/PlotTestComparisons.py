import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


def run(model, color_map, sort_order):
    base_folder = rf"D:\Msc\sem1\DL\AUDIO\Deep_Learning_Projects\Project_2\src\results\{model}"

    csv_files = glob(os.path.join(base_folder, "**", "aggregated_results.csv"), recursive=True)

    hparam_data = {"Experiment": [], "hparam_accuracy": [], "hparam_loss": []}

    for csv_path in csv_files:

        df = pd.read_csv(csv_path)
        print(csv_path)

        # extract params if exist
        accuracy_row = df[df["Metric"] == "hparam_accuracy"]
        loss_row = df[df["Metric"] == "hparam_loss"]

        if not accuracy_row.empty and not loss_row.empty:
            experiment_name = os.path.basename(os.path.dirname(csv_path))
            hparam_data["Experiment"].append(experiment_name)
            hparam_data["hparam_accuracy"].append((accuracy_row["Mean"].values[0], accuracy_row["Std Dev"].values[0]))
            hparam_data["hparam_loss"].append((loss_row["Mean"].values[0], loss_row["Std Dev"].values[0]))

    hparam_df = pd.DataFrame(hparam_data)
    print(hparam_df)

    hparam_df["Sort Key"] = hparam_df["Experiment"].apply(
        lambda x: sort_order.index(x) if x in sort_order else len(sort_order))
    hparam_df = hparam_df.sort_values("Sort Key").drop(columns=["Sort Key"])

    def plot_hparam(metric, title, filename, metric_name):
        experiments = hparam_df["Experiment"]
        metric_means = [val[0] for val in hparam_df[metric]]
        metric_stds = [val[1] for val in hparam_df[metric]]
        colors = [color_map.get(exp, "#7f7f7f") for exp in experiments]  # Default to gray if experiment is unknown

        plt.figure(figsize=(12, 6))
        plt.bar(experiments, metric_means, yerr=metric_stds, capsize=5, alpha=0.7, color=colors)
        plt.xlabel("Architecture")
        plt.ylabel(f"{metric_name} Value")
        plt.ylim(0)
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # save
        plot_path = os.path.join(base_folder, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")

    plot_hparam("hparam_accuracy", f"{model}: Test Accuracy Across Architectures", "hparam_accuracy_plot.png", "Accuracy")
    plot_hparam("hparam_loss", f"{model}: Test Loss Across Architectures", "hparam_loss_plot.png", "Loss")

    hparam_df["hparam_accuracy"] = hparam_df["hparam_accuracy"].apply(lambda x: (float(x[0]), float(x[1])))
    hparam_df["hparam_loss"] = hparam_df["hparam_loss"].apply(lambda x: (float(x[0]), float(x[1])))

    results_path = os.path.join(base_folder, "test_results.csv")
    hparam_df.to_csv(results_path, index=False)

def main():
    color_map = {
        "h4l2": "#a4a4ff",
        "h8l2": "#6d6dff",
        "h16l2": "#3a3aff",
        "h16l3": "#00ff00",
        "h16l1": "#00ff80",
    }
    sort_order = ["h4l2", "h8l2", "h16l2", "h16l3", "h16l1"]

    run("CT", color_map, sort_order)
    color_map["h2l2"] = "#B4B4FF"
    sort_order.insert(0, "h2l2")
    run("TSTF", color_map, sort_order)


if __name__ == '__main__':
    main()