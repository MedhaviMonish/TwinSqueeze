import os
import json
import torch
from model import TwinSqueeze
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from dataloader import load_stsb_data


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


def run_all_model_evaluations(model_dir="models", device="cuda"):
    _, test_loader = load_stsb_data(batch_size=64)
    os.makedirs("results/benchmark_charts", exist_ok=True)

    for fname in os.listdir(model_dir):
        if fname.endswith(".pt"):
            model_path = os.path.join(model_dir, fname)
            model_name = fname.replace(".pt", "")

            # Infer compression from filename
            compressed_dim = 32
            if "cd64" in model_name:
                compressed_dim = 64

            model = TwinSqueeze(compressed_dim=compressed_dim)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            labels, preds, metrics = evaluate_model(model, test_loader, device=device)

            print(f"\n==> {model_name}:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            # Plot predictions vs labels
            plt.figure(figsize=(10, 4))
            plt.plot(labels[:50], label="Ground Truth", color="black", linestyle="--")
            plt.plot(preds[:50], label="Predicted", alpha=0.8)
            plt.title(f"{model_name} Predictions vs Ground Truth")
            plt.xlabel("Sample Index")
            plt.ylabel("Cosine Similarity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/benchmark_charts/{model_name}_pred_vs_actual.png")
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