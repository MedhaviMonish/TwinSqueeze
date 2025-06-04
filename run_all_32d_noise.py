import subprocess

alphas = [0.0, 0.5, 0.75, 1.0, 3.0]  # baseline and NEFTune variants
compressed_dim = 32
epochs = 200

for alpha in alphas:
    cmd = [
        "python", "run_experiment.py",
        "--compressed_dim", str(compressed_dim),
        "--alpha", str(alpha),
        "--epochs", str(epochs)
    ]
    print(f"\n🔧 Running: alpha={alpha}, dim={compressed_dim}, epochs={epochs}")
    subprocess.run(cmd)
