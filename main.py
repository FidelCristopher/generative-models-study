import argparse
import torch
from exp_mlp_limits import run_mlp_experiments
from train_generative import train_all_models
from evaluate_and_plot import run_evaluation_and_figures

def main():
    parser = argparse.ArgumentParser(description="Studi Komparatif Model Generatif")
    parser.add_argument("--step", type=str, default="all", choices=["mlp", "train", "eval", "all"],
                        help="Tahap eksekusi: mlp, train, eval, all")
    parser.add_argument("--epochs", type=int, default=15, help="Jumlah epoch (default: 15)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Menggunakan device: {device}")

    if args.step in ["mlp", "all"]:
        print("Menjalankan eksperimen batasan MLP classifier...")
        run_mlp_experiments(device=device)

    if args.step in ["train", "all"]:
        print(f"Training 6 model generatif ({args.epochs} epoch)...")
        train_all_models(device=device, epochs=args.epochs)

    if args.step in ["eval", "all"]:
        print("Evaluasi metrik dan plotting figur visual...")
        run_evaluation_and_figures(device=device)

    print("Seluruh proses eksperimen selesai.")

if __name__ == "__main__":
    main()
