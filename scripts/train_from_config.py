"""Universal training script for DeepDPM experiments.

Usage:
    python train_from_config.py \\
        --config path/to/config.yaml \\
        --embeddings path/to/embeddings.pt \\
        --labels path/to/labels.pt \\
        --name "Experiment Name"
"""

import sys
import argparse
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
from data import load_embeddings_from_file, create_dataloader
from training import DeepDPMTrainer
from configs import TrainingConfig
from utils import set_seed


def main():
    """Train DeepDPM with specified config and data."""

    parser = argparse.ArgumentParser(
        description="Train DeepDPM clustering model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to embeddings .pt file"
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels .pt file"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Experiment name for display"
    )

    args = parser.parse_args()

    # Convert to absolute paths
    config_path = Path(args.config).resolve()
    embeddings_path = Path(args.embeddings).resolve()
    labels_path = Path(args.labels).resolve()

    # Validate paths exist
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    if not embeddings_path.exists():
        print(f"Error: Embeddings file not found: {embeddings_path}")
        return 1
    if not labels_path.exists():
        print(f"Error: Labels file not found: {labels_path}")
        return 1

    print("=" * 80)
    print(f"{args.name.upper()} EXPERIMENT")
    print("=" * 80)
    print()

    # Load configuration from YAML
    print("[1/5] Loading configuration...")
    config = TrainingConfig.from_yaml(str(config_path))
    print(f"✓ Config loaded from {config_path.name}")

    print()

    # Load data
    print("[2/5] Loading embeddings...")
    data, labels = load_embeddings_from_file(
        str(embeddings_path),
        str(labels_path)
    )
    print(f"✓ Data shape: {data.shape}")
    print(f"✓ True number of clusters: {len(torch.unique(labels))}")
    print()

    # Verify input dimension matches config
    if data.shape[1] != config.input_dim:
        raise ValueError(
            f"Data dimension mismatch: expected {config.input_dim}, "
            f"got {data.shape[1]}"
        )

    # Set seed
    set_seed(config.seed)

    # Create dataloader
    print("[3/5] Creating dataloader...")
    train_loader = create_dataloader(
        data=data,
        labels=labels,
        batch_size=config.batch_size,
        shuffle=True
    )
    print(f"✓ Dataloader created: {len(train_loader)} batches")
    print()

    # Initialize trainer
    print("[4/5] Initializing trainer...")
    trainer = DeepDPMTrainer(config)
    print("✓ Trainer initialized")
    print()

    # Train
    print("[5/5] Training...")
    print("-" * 80)
    trainer.train(train_loader, true_labels=labels)
    print("-" * 80)
    print()

    # Final results
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"True K: {len(torch.unique(labels))}")
    print(f"Inferred K: {trainer.gmm_params.k}")

    if trainer.training_history:
        final_metrics = trainer.training_history[-1]
        print(f"Final Cluster Loss: {final_metrics['cluster_loss']:.4f}")
        if 'subcluster_loss' in final_metrics:
            print(f"Final Subcluster Loss: {final_metrics['subcluster_loss']:.4f}")
        print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Final NMI: {final_metrics['nmi']:.4f}")
        print(f"Final ARI: {final_metrics['ari']:.4f}")

    print("=" * 80)

    # Save final checkpoint
    final_checkpoint_path = Path(config.checkpoint_dir) / "final_model.pt"
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"\nFinal model saved to {final_checkpoint_path}")

    # Save results
    results_path = Path(config.log_dir) / "final_results.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write(f"{args.name} Results\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"True K: {len(torch.unique(labels))}\n")
        f.write(f"Inferred K: {trainer.gmm_params.k}\n")
        if trainer.training_history:
            final = trainer.training_history[-1]
            f.write(f"Accuracy: {final['accuracy']:.4f}\n")
            f.write(f"NMI: {final['nmi']:.4f}\n")
            f.write(f"ARI: {final['ari']:.4f}\n")

    print(f"Results saved to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
