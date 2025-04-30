import json
import os
from glob import glob

import numpy as np
import pandas as pd


def run(model, scenario):
    results_folder = rf"D:\Msc\sem1\DL\AUDIO\Deep_Learning_Projects\Project_2\src\results\{model}\{scenario}"

    json_files = glob(os.path.join(results_folder, "*.json"))

    data = {}

    for file_path in json_files:
        filename = os.path.basename(file_path)
        parts = filename.split("-tag-")

        if len(parts) < 2:
            continue  # skip unexpected filenames

        metric_type = parts[1].replace(".json", "")  # extract metric name

        with open(file_path, "r") as f:
            json_data = json.load(f)

        filtered_values = [(entry[1], entry[2]) for entry in json_data if isinstance(entry, list) and len(entry) >= 3]

        if metric_type not in data:
            data[metric_type] = {}

        for epoch, metric_value in filtered_values:
            if epoch not in data[metric_type]:
                data[metric_type][epoch] = []

            data[metric_type][epoch].append(metric_value)

    results = []

    for metric, epochs in data.items():
        for epoch, values in epochs.items():
            mean_value = np.mean(values)
            std_value = np.std(values)

            results.append({"Metric": metric, "Epoch": epoch, "Mean": mean_value, "Std Dev": std_value})

    df = pd.DataFrame(results)
    print(df.head())

    # save
    csv_path = os.path.join(results_folder, "aggregated_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"Aggregated results saved to: {csv_path}")


def main():
    experiments = {
        "CT": ["h4l2", "h8l2", "h16l2", "h16l3", "h16l1"],
        "TSTF": ["h2l2", "h4l2", "h8l2", "h16l2", "h16l3", "h16l1"]
    }
    for model in experiments.keys():
        for scenario in experiments[model]:
            run(model, scenario)


if __name__ == '__main__':
    main()
