import torch
import torch.nn as nn
from model import TwinSqueeze
from trainer import TwinSqueezeTrainer
from logger import ExperimentLogger
from dataloader import load_stsb_data
import argparse


def run_experiment(compressed_dim=32, alpha=0.0, epochs=50, batch_size=32, lr=2e-4, device="cpu"):
    print(f"Running on `{device}` device")
    model_name = f"twin_cd{compressed_dim}"+ ("_baseline" if alpha == 0.0 else f"_alpha{alpha}")

    # Load data
    train_dataloader, test_dataloader = load_stsb_data(batch_size=batch_size, alpha=0.0)

    # Init model and optimizer
    model = TwinSqueeze(compressed_dim=compressed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Trainer
    trainer = TwinSqueezeTrainer(model, optimizer, criterion, device)
    trainer.train(train_dataloader, epochs=epochs)

    # Logging
    logger = ExperimentLogger()
    logger.save_model_path = f"models/{model_name}.pt"
    trainer.save_model(logger.save_model_path)
    logger.save_loss_curve(trainer.get_loss_log(), model_name)
    logger.save_loss_log_json(trainer.get_loss_log(), model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed_dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_experiment(**vars(args))