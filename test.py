import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn.functional as F
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

def compute_per_sample_losses(model, dataloader, device="cpu"):
    model.eval()
    losses = []
    with torch.no_grad():
        for emb1, emb2, labels in dataloader:
            emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device)
            preds = model(emb1, emb2)
            per_example_loss = F.mse_loss(preds, labels, reduction='none')
            losses.extend(per_example_loss.cpu().numpy())
    return losses

def generate_loss_overlay_per_alpha():
    device = "cpu"
    train_loader, test_loader = load_stsb_data(batch_size=64)

    base_model = TwinSqueeze()
    base_model.load_state_dict(torch.load("models/twin_cd32_baseline.pt", map_location=device))
    base_train = compute_per_sample_losses(base_model, train_loader, device)
    base_test = compute_per_sample_losses(base_model, test_loader, device)

    alpha_variants = {
        "0.5": "twin_cd32_alpha0.5",
        "0.75": "twin_cd32_alpha0.75",
        "1.0": "twin_cd32_alpha1.0",
        "3.0": "twin_cd32_alpha3.0"
    }

    for alpha, model_name in alpha_variants.items():
        model = TwinSqueeze()
        model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=device))
        train_losses = compute_per_sample_losses(model, train_loader, device)
        test_losses = compute_per_sample_losses(model, test_loader, device)

        plt.figure(figsize=(12, 5))

        # Train Loss Comparison
        plt.subplot(1, 2, 1)
        plt.yscale("log")
        plt.hist(base_train, bins=50, alpha=0.6, label="Baseline", color="steelblue", edgecolor="black")
        plt.hist(train_losses, bins=50, alpha=0.6, label=f"α = {alpha}", color="orange", edgecolor="black")
        plt.xlabel("Loss (Train)")
        plt.ylabel("Log Count")
        plt.title(f"Train Loss: Baseline vs α = {alpha}")
        plt.legend()

        # Test Loss Comparison
        plt.subplot(1, 2, 2)
        plt.yscale("log")
        plt.hist(base_test, bins=50, alpha=0.6, label="Baseline", color="steelblue", edgecolor="black")
        plt.hist(test_losses, bins=50, alpha=0.6, label=f"α = {alpha}", color="orange", edgecolor="black")
        plt.xlabel("Loss (Test)")
        plt.ylabel("Log Count")
        plt.title(f"Test Loss: Baseline vs α = {alpha}")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"results/benchmark_charts/loss_compare_alpha_{alpha}.png")
        plt.close()

if __name__ == "__main__":
    run_all_model_evaluations()           
    plot_loss_charts() 
    generate_loss_overlay_per_alpha()