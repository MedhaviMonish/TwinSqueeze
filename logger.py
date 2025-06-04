import json
import csv
import os
import matplotlib.pyplot as plt

class ExperimentLogger:
    def __init__(self, result_dir="results"):
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

    def save_loss_curve(self, loss_log, model_name):
        plt.figure()
        plt.plot(loss_log)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve - {model_name}")
        plt.grid(True)
        path = os.path.join(self.result_dir, f"loss_{model_name}.png")
        plt.savefig(path)
        plt.close()

    def save_metrics_csv(self, metrics_dict, dataset_name):
        csv_path = os.path.join(self.result_dir, f"metrics_{dataset_name}.csv")
        header = ["model", "alpha", "compressed_dim", "MSE", "MAE", "Pearson", "Spearman"]
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([metrics_dict.get(k, "") for k in header])

    def save_loss_log_json(self, loss_log, model_name):
        path = os.path.join(self.result_dir, f"loss_log_{model_name}.json")
        with open(path, "w") as f:
            json.dump(loss_log, f, indent=2)