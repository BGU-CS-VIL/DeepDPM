"""Training script for synthetic data."""

import sys
from pathlib import Path

# Add repo root to path for package imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
from data import create_synthetic_gmm_data, create_dataloader
from training import DeepDPMTrainer
from configs import TrainingConfig
from utils import set_seed


def main():
    """Train DeepDPM on synthetic GMM data."""

    # Configuration
    config = TrainingConfig(
        # Data
        input_dim=10,

        # Model
        hidden_dims=[50],
        init_k=1,

        # Training
        batch_size=128,
        num_epochs=200,
        cluster_lr=0.0005,
        subcluster_lr=0.005,

        # Loss
        cluster_loss_type="KL_GMM_2",
        subcluster_loss_type="isotropic",

        # Schedule
        gmm_warmup_epochs=20,
        start_sub_clustering=25,
        start_splitting=35,
        start_merging=35,
        split_merge_every_n_epochs=20,
        eval_every_n_epochs=10,

        # Reproducibility
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",

        # Logging
        log_dir="./logs/synthetic",
        checkpoint_dir="./checkpoints/synthetic",
        save_every_n_epochs=50,
        verbose=True
    )

    # Set seed
    set_seed(config.seed)

    # Create synthetic data
    print("Creating synthetic GMM data...")
    data, labels = create_synthetic_gmm_data(
        n_samples=2000,
        n_features=config.input_dim,
        n_clusters=5,
        cluster_std=1.0,
        random_state=config.seed
    )

    print(f"Data shape: {data.shape}")
    print(f"True number of clusters: {len(torch.unique(labels))}")

    # Create dataloader
    train_loader = create_dataloader(
        data,
        labels=None,  # Don't use labels during training
        batch_size=config.batch_size,
        shuffle=True
    )

    # Initialize trainer
    trainer = DeepDPMTrainer(config)

    # Train
    trainer.train(train_loader, true_labels=labels)

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"True K: {len(torch.unique(labels))}")
    print(f"Inferred K: {trainer.gmm_params.k}")

    if trainer.training_history:
        final_metrics = trainer.training_history[-1]
        print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Final NMI: {final_metrics['nmi']:.4f}")
        print(f"Final ARI: {final_metrics['ari']:.4f}")

    print("=" * 70)

    # Save final checkpoint
    final_checkpoint_path = Path(config.checkpoint_dir) / "final_model.pt"
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"\nFinal model saved to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
