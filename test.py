import os
import numpy as np
import json
import torch
from model import TwinSqueeze
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from dataloader import load_stsb_data
from logger import ExperimentLogger  # Make sure it's imported

logger = ExperimentLogger()

def evaluate_model(model, dataloader, device="cpu", sample_plot=30):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for emb1, emb2, labels in dataloader:
            emb1, emb2 = emb1.to(device), emb2.to(device)
            preds = model(emb1, emb2).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    pearson = pearsonr(all_labels, all_preds)[0]
    spearman = spearmanr(all_labels, all_preds)[0]

    return all_labels, all_preds, {
        "MSE": mse,
        "MAE": mae,
        "Pearson": pearson,
        "Spearman": spearman
    }

def run_all_model_evaluations(model_dir="models", device="cuda", step=30):
    _, test_loader = load_stsb_data(batch_size=64)
    os.makedirs("results/benchmark_charts", exist_ok=True)

    from logger import ExperimentLogger
    logger = ExperimentLogger()

    all_preds_dict = {}
    labels_ref = None

    for fname in os.listdir(model_dir):
        if fname.endswith(".pt"):
            model_path = os.path.join(model_dir, fname)
            model_name = fname.replace(".pt", "")

            compressed_dim = 64 if "cd64" in model_name else 32
            model = TwinSqueeze(compressed_dim=compressed_dim)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            labels, preds, metrics = evaluate_model(model, test_loader, device=device)
            all_preds_dict[model_name] = preds
            if labels_ref is None:
                labels_ref = labels  # Store once

            print(f"\n==> {model_name}:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            # ✅ Save individual model vs ground truth (just this model)
            down_labels = labels[::step]
            down_preds = preds[::step]
            plt.figure(figsize=(12, 4))
            plt.plot(down_labels, label="Ground Truth", linestyle="--", color="black")
            plt.plot(down_preds, label="Predicted", alpha=0.9)
            plt.title(f"{model_name} vs Ground Truth")
            plt.xlabel("Sample Index (downsampled)")
            plt.ylabel("Cosine Similarity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/benchmark_charts/{model_name}_vs_truth.png")
            plt.close()

            # ✅ Save to metrics CSV
            logger.save_metrics_csv(
                metrics_dict={
                    "model": model_name,
                    "alpha": model_name.split("alpha")[-1] if "alpha" in model_name else "0.0",
                    "compressed_dim": compressed_dim,
                    **metrics
                },
                dataset_name="stsb"
            )

    # === Create comparison plots: each alpha vs baseline + ground truth ===
    base_preds = all_preds_dict.get("twin_cd32_baseline")
    if base_preds is None:
        print("⚠️ Baseline model 'twin_cd32_baseline' not found in models/")
        return

    down_labels = labels_ref[::step]
    down_base = base_preds[::step]

    for model_name, preds in all_preds_dict.items():
        if model_name == "twin_cd32_baseline":
            continue  # skip baseline alone

        down_preds = preds[::step]

        plt.figure(figsize=(12, 4))
        plt.plot(down_labels, label="Ground Truth", linestyle="--", color="black")
        plt.plot(down_base, label="Baseline", linestyle=":", color="blue")
        plt.plot(down_preds, label=model_name, alpha=0.9)
        plt.title(f"{model_name} vs Baseline vs Ground Truth")
        plt.xlabel("Sample Index (downsampled)")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/benchmark_charts/{model_name}_vs_baseline.png")
        plt.close()

    # === Combined comparison plot ===
    plt.figure(figsize=(14, 6))
    plt.plot(down_labels, label="Ground Truth", linestyle="--", color="black", linewidth=2)

    for model_name, preds in all_preds_dict.items():
        plt.plot(preds[::step], label=model_name, alpha=0.8)

    plt.title("All Models: Cosine Predictions vs Ground Truth")
    plt.xlabel("Sample Index (downsampled)")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/benchmark_charts/all_models_comparison.png")
    plt.close()


def plot_loss_charts():
    # Load loss logs from results folder
    results_dir = "results"
    loss_logs = {}

    for file in os.listdir(results_dir):
        if file.startswith("loss_log_") and file.endswith(".json"):
            model_name = file.replace("loss_log_", "").replace(".json", "")
            path = os.path.join(results_dir, file)
            with open(path, "r") as f:
                loss_logs[model_name] = json.load(f)

    # Plot all losses
    plt.figure(figsize=(12, 6))
    for model_name, loss_list in loss_logs.items():
        plt.plot(loss_list, label=model_name)

    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/loss_comparison.png")
    plt.close()

if __name__ == "__main__":
    run_all_model_evaluations()           
    plot_loss_charts() 