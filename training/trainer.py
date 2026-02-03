"""Main training loop for DeepDPM."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
from enum import Enum, auto
import time

# Module logger
logger = logging.getLogger(__name__)

# Threshold for detecting degenerate subcluster splits (>90% or <10% imbalance)
DEGENERACY_THRESHOLD = 0.1


class SplitMergeAction(Enum):
    """Enum for tracking the last split/merge action performed."""
    SPLIT = auto()
    MERGE = auto()

from models import ClusterNet, SubclusterNet
from gmm import GMMParameters, NIWPrior, set_kmeans_seed
from losses import kl_gmm_loss, isotropic_loss, subcluster_isotropic_loss
from split_merge import propose_splits, propose_merges, set_split_seed, set_merge_seed
from split_merge.split import SplitConfig
from split_merge.merge import MergeConfig
from configs import TrainingConfig
from training.metrics import compute_clustering_metrics


class DeepDPMTrainer:
    """Main training loop for DeepDPM clustering.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Models (initialized in train())
        self.cluster_net: Optional[ClusterNet] = None
        self.subcluster_net: Optional[SubclusterNet] = None
        self.gmm_params: Optional[GMMParameters] = None
        self.prior: Optional[NIWPrior] = None

        # Optimizers (initialized in train())
        self.cluster_optimizer = None
        self.subcluster_optimizer = None

        # Training state
        self.current_epoch = 0
        self.training_history = []

        # Track when splits/merges occurred (for parameter freezing)
        # GMM params are frozen for N epochs after split/merge
        self.last_split_epoch: Optional[int] = None
        self.last_merge_epoch: Optional[int] = None

        # Alternation state for split/merge
        # Initialized to MERGE to allow first split at start_splitting epoch
        self.last_performed: SplitMergeAction = SplitMergeAction.MERGE

        # Create directories
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize RNG seeds for reproducibility
        if hasattr(config, 'seed') and config.seed is not None:
            set_split_seed(config.seed)
            set_merge_seed(config.seed)
            set_kmeans_seed(config.seed)
            logger.debug(f"Initialized split/merge/kmeans RNG with seed {config.seed}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        true_labels: Optional[torch.Tensor] = None
    ) -> None:
        """Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            true_labels: Optional ground truth labels for evaluation
        """
        logger.info("=" * 70)
        logger.info("DeepDPM Training")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Initial K: {self.config.init_k}")
        logger.info(f"Num epochs: {self.config.num_epochs}")
        logger.info("=" * 70)

        # Epoch 0: Initialize GMM parameters using K-means
        if self.gmm_params is None:
            self._initialize_gmm(train_loader)

        # Training loop
        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            split_performed_this_epoch = False
            merge_performed_this_epoch = False

            # Train both networks on the same data pass
            cluster_loss, subcluster_loss = self._train_epoch(
                train_loader,
                train_subcluster=(epoch >= self.config.start_sub_clustering)
            )

            # Update learning rate schedulers (skip during freeze period)
            scheduler_should_skip = self._should_freeze_gmm_params()

            if self.config.use_cluster_lr_scheduler and self.cluster_scheduler is not None:
                if not scheduler_should_skip:
                    self.cluster_scheduler.step(cluster_loss)

            if self.config.use_subcluster_lr_scheduler and self.subcluster_scheduler is not None:
                if epoch >= self.config.start_sub_clustering and not scheduler_should_skip:
                    self.subcluster_scheduler.step(subcluster_loss)

            # Update GMM parameters (M-step)
            should_freeze = self._should_freeze_gmm_params()

            if epoch >= self.config.gmm_warmup_epochs and not should_freeze:
                self._update_gmm_parameters(train_loader)
                # Log imbalanced subclusters
                if epoch >= self.config.start_sub_clustering:
                    degenerate_clusters = []
                    for k in range(self.gmm_params.k):
                        pi_sub_0 = self.gmm_params.pi_sub[2*k].item()
                        pi_sub_1 = self.gmm_params.pi_sub[2*k+1].item()
                        total = pi_sub_0 + pi_sub_1
                        if total > 0:
                            ratio = pi_sub_0 / total
                            if ratio < DEGENERACY_THRESHOLD or ratio > (1 - DEGENERACY_THRESHOLD):
                                degenerate_clusters.append((k, ratio))
                    if degenerate_clusters:
                        logger.warning(f"Imbalanced subclusters: {[(k, f'{r:.2f}') for k, r in degenerate_clusters]}")
            elif should_freeze:
                logger.debug(f"GMM frozen (within {self.config.freeze_mus_after_split_merge} epochs of split/merge)")

            # Split/Merge proposals
            num_splits, num_merges = 0, 0

            # Compute split/merge eligibility
            perform_split = (
                epoch >= self.config.start_splitting
                and (epoch - self.config.start_splitting) % self.config.split_merge_every_n_epochs == 0
                and self.last_performed == SplitMergeAction.MERGE
            )

            perform_merge = (
                epoch >= self.config.start_merging
                and (epoch - self.config.start_merging) % self.config.split_merge_every_n_epochs == 0
                and not split_performed_this_epoch
                and self.last_performed == SplitMergeAction.SPLIT
            )

            # Execute splits
            if perform_split:
                self.last_performed = SplitMergeAction.SPLIT
                num_splits = self._execute_splits(train_loader)
                if num_splits > 0:
                    split_performed_this_epoch = True
                    self._reset_scheduler_state()

            # Execute merges
            if perform_merge:
                self.last_performed = SplitMergeAction.MERGE
                num_merges = self._execute_merges(train_loader)
                if num_merges > 0:
                    merge_performed_this_epoch = True
                    self._reset_scheduler_state()

            # Evaluation and logging
            epoch_time = time.time() - epoch_start_time

            if epoch % self.config.eval_every_n_epochs == 0 or epoch == self.config.num_epochs:
                metrics = self._evaluate(train_loader, true_labels)
                self._log_epoch(epoch, cluster_loss, subcluster_loss,
                               num_splits, num_merges, metrics, epoch_time)

                # Save history
                self.training_history.append({
                    'epoch': epoch,
                    'K': self.gmm_params.k,
                    'cluster_loss': cluster_loss,
                    'subcluster_loss': subcluster_loss,
                    'num_splits': num_splits,
                    'num_merges': num_merges,
                    **metrics
                })

                # Prune history if exceeded limit (keep early + recent entries)
                max_size = self.config.max_history_size
                if max_size > 0 and len(self.training_history) > max_size:
                    keep_early = max(1, max_size // 10)  # 10% from early training
                    keep_recent = max_size - keep_early
                    self.training_history = (
                        self.training_history[:keep_early] +
                        self.training_history[-keep_recent:]
                    )
                    logger.debug(f"Pruned training_history to {len(self.training_history)} entries")
            else:
                # Brief logging
                logger.info(f"Epoch {epoch:3d}/{self.config.num_epochs}: "
                            f"K={self.gmm_params.k:2d}, "
                            f"cluster_loss={cluster_loss:.4f}, "
                            f"sub_loss={subcluster_loss:.4f}, "
                            f"splits={num_splits}, merges={num_merges}, "
                            f"time={epoch_time:.1f}s")

            # Save checkpoint
            if epoch % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(
                    Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
                )

        logger.info("=" * 70)
        logger.info("Training completed!")
        logger.info(f"Final K: {self.gmm_params.k}")
        logger.info("=" * 70)

    def _initialize_gmm(self, dataloader: DataLoader) -> None:
        """Initialize GMM parameters using K-means."""
        logger.info("Gathering data and initializing with K-means...")

        # Gather all data from the underlying dataset (in dataset order)
        # This avoids the shuffling issue when comparing with true_labels
        dataset = dataloader.dataset
        all_data_list = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, (list, tuple)):
                data = item[0]
            else:
                data = item
            all_data_list.append(data.unsqueeze(0) if data.dim() == 1 else data)

        all_data = torch.cat(all_data_list, dim=0).to(self.device)

        # Store for evaluation (in dataset order, matching true_labels)
        self.all_data = all_data
        N, d = all_data.shape

        logger.info(f"Data shape: {all_data.shape}")
        logger.info(f"Initializing with K={self.config.init_k}")

        # Create prior
        self.prior = NIWPrior.from_data(
            all_data,
            kappa=self.config.prior_config.kappa,
            nu_offset=self.config.prior_config.nu_offset,
            psi_scale=self.config.prior_config.psi_scale
        )

        # Initialize GMM parameters
        self.gmm_params = GMMParameters.initialize_from_kmeans(
            all_data,
            self.config.init_k,
            self.prior
        )

        # Initialize networks
        self.cluster_net = ClusterNet(
            input_dim=d,
            hidden_dims=self.config.hidden_dims,
            init_k=self.config.init_k,
            softmax_norm=self.config.cluster_softmax_norm
        ).to(self.device)

        self.subcluster_net = SubclusterNet(
            input_dim=d,
            init_k=self.config.init_k,
            softmax_norm=self.config.subcluster_softmax_norm
        ).to(self.device)

        # Initialize optimizers with separate param_groups for momentum preservation
        encoder_params = list(self.cluster_net.hidden_layers.parameters())
        self.cluster_optimizer = torch.optim.Adam(
            encoder_params,
            lr=self.config.cluster_lr
        )
        # Add output layer as separate param group
        self.cluster_optimizer.add_param_group(
            {"params": self.cluster_net.output_layer.parameters()}
        )

        self.subcluster_optimizer = torch.optim.Adam(
            self.subcluster_net.parameters(),
            lr=self.config.subcluster_lr
        )

        # Initialize learning rate schedulers
        if self.config.use_cluster_lr_scheduler:
            self.cluster_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.cluster_optimizer,
                mode=self.config.lr_scheduler_mode,
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience
            )
        else:
            self.cluster_scheduler = None

        if self.config.use_subcluster_lr_scheduler:
            self.subcluster_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.subcluster_optimizer,
                mode=self.config.lr_scheduler_mode,
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience
            )
        else:
            self.subcluster_scheduler = None

        logger.info(f"Initialization complete. K={self.gmm_params.k}")
        if self.config.use_cluster_lr_scheduler:
            logger.info(f"ClusterNet LR Scheduler: ReduceLROnPlateau (patience={self.config.lr_scheduler_patience}, factor={self.config.lr_scheduler_factor})")
        if self.config.use_subcluster_lr_scheduler:
            logger.info(f"SubclusterNet LR Scheduler: ReduceLROnPlateau (patience={self.config.lr_scheduler_patience}, factor={self.config.lr_scheduler_factor})")

    def _train_epoch(
        self,
        dataloader: DataLoader,
        train_subcluster: bool = True
    ) -> tuple[float, float]:
        """Train networks for one epoch.

        Trains ClusterNet and optionally SubclusterNet on the same data pass,
        ensuring both networks see the same batches (avoiding shuffle mismatch).

        Args:
            dataloader: Training data loader
            train_subcluster: Whether to train SubclusterNet this epoch

        Returns:
            (cluster_loss, subcluster_loss) averaged over epoch
        """
        self.cluster_net.train()
        if train_subcluster:
            self.subcluster_net.train()

        total_cluster_loss = 0.0
        total_subcluster_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch

            data = data.to(self.device)

            # === Phase 1: Train ClusterNet ===
            cluster_probs = self.cluster_net(data)

            if self.config.cluster_loss_type == "KL_GMM_2":
                cluster_loss = kl_gmm_loss(cluster_probs, self.gmm_params, data)
            elif self.config.cluster_loss_type == "isotropic":
                cluster_loss = isotropic_loss(cluster_probs, self.gmm_params, data)
            else:
                raise ValueError(f"Unknown cluster loss type: {self.config.cluster_loss_type}")

            cluster_loss = cluster_loss * self.config.cluster_loss_weight

            self.cluster_optimizer.zero_grad()
            cluster_loss.backward()
            self.cluster_optimizer.step()

            total_cluster_loss += cluster_loss.item()

            # Train SubclusterNet (on same batch)
            if train_subcluster:
                with torch.no_grad():
                    cluster_probs_detached = self.cluster_net(data)
                    cluster_assignments = cluster_probs_detached.argmax(dim=1)

                subcluster_resp = self._compute_subcluster_resp(data, cluster_assignments)
                subcluster_loss = subcluster_isotropic_loss(
                    subcluster_resp,  # Already softmaxed with masking
                    self.gmm_params,
                    data,
                    cluster_assignments
                )
                subcluster_loss = subcluster_loss * self.config.subcluster_loss_weight

                self.subcluster_optimizer.zero_grad()
                subcluster_loss.backward()
                self.subcluster_optimizer.step()

                total_subcluster_loss += subcluster_loss.item()

            num_batches += 1

        avg_cluster_loss = total_cluster_loss / num_batches
        avg_subcluster_loss = total_subcluster_loss / num_batches if train_subcluster else 0.0

        return avg_cluster_loss, avg_subcluster_loss

    def _compute_subcluster_resp(
        self,
        data: torch.Tensor,
        cluster_assignments: torch.Tensor
    ) -> torch.Tensor:
        """Compute subcluster responsibilities with masked softmax.

        Args:
            data: (N, d) input features
            cluster_assignments: (N,) hard cluster assignments

        Returns:
            (N, 2*K) subcluster responsibilities (softmaxed)
        """
        N = data.size(0)
        K = self.gmm_params.k

        subcluster_logits = self.subcluster_net(data)

        # Create mask for relevant subclusters
        mask = torch.zeros(N, 2 * K, device=data.device)
        batch_indices = torch.arange(N, device=data.device)
        mask[batch_indices, 2 * cluster_assignments] = 1.0
        mask[batch_indices, 2 * cluster_assignments + 1] = 1.0

        # Apply masked softmax with temperature scaling
        logits_masked = subcluster_logits.masked_fill((1 - mask).bool(), float('-inf'))
        subcluster_resp = F.softmax(logits_masked * self.subcluster_net.softmax_norm, dim=1)

        return subcluster_resp

    def _should_freeze_gmm_params(self) -> bool:
        """Check if GMM parameters should be frozen.

        GMM params are frozen for N epochs after split/merge to let the
        networks stabilize before updating GMM parameters.
        """
        freeze_after_split = (
            self.last_split_epoch is not None and
            (self.current_epoch - self.last_split_epoch) <= self.config.freeze_mus_after_split_merge
        )
        freeze_after_merge = (
            self.last_merge_epoch is not None and
            (self.current_epoch - self.last_merge_epoch) <= self.config.freeze_mus_after_split_merge
        )
        return freeze_after_split or freeze_after_merge

    def _update_gmm_parameters(self, dataloader: DataLoader) -> None:
        """Update GMM parameters using M-step with network responsibilities."""
        if self._should_freeze_gmm_params():
            logger.debug("GMM parameters frozen (recent split/merge)")
            return

        self.cluster_net.eval()
        self.subcluster_net.eval()

        # Gather all data and responsibilities
        all_data = []
        all_cluster_resp = []
        all_cluster_assignments = []
        all_subcluster_resp = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch

                data = data.to(self.device)

                # Cluster responsibilities
                cluster_probs = self.cluster_net(data)
                cluster_assignments = cluster_probs.argmax(dim=1)

                subcluster_probs = self._compute_subcluster_resp(data, cluster_assignments)

                all_data.append(data)
                all_cluster_resp.append(cluster_probs)
                all_cluster_assignments.append(cluster_assignments)
                all_subcluster_resp.append(subcluster_probs)

        all_data = torch.cat(all_data, dim=0)
        all_cluster_resp = torch.cat(all_cluster_resp, dim=0)
        all_cluster_assignments = torch.cat(all_cluster_assignments, dim=0)
        all_subcluster_resp = torch.cat(all_subcluster_resp, dim=0)

        # M-step: Update cluster parameters
        self.gmm_params.update_from_responsibilities(
            all_data,
            all_cluster_resp,
            self.prior
        )

        # Update subcluster parameters (with warmup period)
        # Wait for subcluster net to train before updating GMM subclusters
        subcluster_update_start = (
            self.config.start_sub_clustering +
            self.config.subcluster_warmup_epochs
        )
        if self.current_epoch >= subcluster_update_start:
            self.gmm_params.update_subclusters_from_assignments(
                all_data,
                all_cluster_assignments,
                all_subcluster_resp,
                self.prior
            )

    def _update_optimizers_after_architecture_change(self) -> None:
        """Update optimizer param_groups after network architecture changes.

        ClusterNet: Clears optimizer state for output layer only, preserves encoder momentum.
        SubclusterNet: Clears optimizer state for all parameters.
        """
        logger.debug("Updating optimizer param_groups after architecture change")

        # ClusterNet: Clear state for output layer only
        encoder_param_ids = {id(p) for p in self.cluster_net.hidden_layers.parameters()}
        for p in list(self.cluster_optimizer.state.keys()):
            if id(p) not in encoder_param_ids:
                self.cluster_optimizer.state.pop(p)
        self.cluster_optimizer.param_groups[1]["params"] = list(
            self.cluster_net.output_layer.parameters()
        )

        # SubclusterNet: Clear state for all parameters
        for p in list(self.subcluster_optimizer.state.keys()):
            self.subcluster_optimizer.state.pop(p)
        self.subcluster_optimizer.param_groups[0]["params"] = list(
            self.subcluster_net.parameters()
        )

    def _reset_scheduler_state(self) -> None:
        """Reset scheduler internal state after architecture changes.

        After a split/merge, the loss landscape changes completely and the
        scheduler's tracked history is no longer relevant.
        """
        if self.cluster_scheduler is not None:
            # Reset ReduceLROnPlateau internal state
            if hasattr(self.cluster_scheduler, 'best'):
                self.cluster_scheduler.best = float('inf') if self.cluster_scheduler.mode == 'min' else float('-inf')
            if hasattr(self.cluster_scheduler, 'num_bad_epochs'):
                self.cluster_scheduler.num_bad_epochs = 0
            if hasattr(self.cluster_scheduler, 'cooldown_counter'):
                self.cluster_scheduler.cooldown_counter = 0

        if self.subcluster_scheduler is not None:
            if hasattr(self.subcluster_scheduler, 'best'):
                self.subcluster_scheduler.best = float('inf') if self.subcluster_scheduler.mode == 'min' else float('-inf')
            if hasattr(self.subcluster_scheduler, 'num_bad_epochs'):
                self.subcluster_scheduler.num_bad_epochs = 0
            if hasattr(self.subcluster_scheduler, 'cooldown_counter'):
                self.subcluster_scheduler.cooldown_counter = 0

    def _execute_splits(self, dataloader: DataLoader) -> int:
        """Execute split proposals.

        Returns:
            Number of splits accepted
        """
        # Gather all data and subcluster assignments
        all_data = []
        all_subcluster_assignments = []
        all_cluster_assignments = []

        self.cluster_net.eval()
        self.subcluster_net.eval()

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch

                data = data.to(self.device)

                cluster_probs = self.cluster_net(data)
                cluster_assignments = cluster_probs.argmax(dim=1)

                # Get all subcluster probabilities
                subcluster_probs_all = self.subcluster_net.forward_all_subclusters(data)

                all_data.append(data)
                all_subcluster_assignments.append(subcluster_probs_all)
                all_cluster_assignments.append(cluster_assignments)

        all_data = torch.cat(all_data, dim=0)
        all_subcluster_assignments = torch.cat(all_subcluster_assignments, dim=0)
        all_cluster_assignments = torch.cat(all_cluster_assignments, dim=0)

        # Propose splits
        split_config = SplitConfig(
            alpha=self.config.split_config.alpha,
            stochastic_accept=self.config.split_config.stochastic_accept,
            min_cluster_size=self.config.split_config.min_cluster_size,
            ignore_subclusters=self.config.split_config.ignore_subclusters
        )

        splits = propose_splits(
            self.gmm_params,
            all_data,
            all_subcluster_assignments,
            all_cluster_assignments,
            self.prior,
            split_config
        )

        for cluster_idx in splits:
            self.gmm_params.split_cluster(
                cluster_idx,
                all_data,
                all_subcluster_assignments,
                self.prior,
                ignore_subclusters=self.config.split_config.ignore_subclusters
            )
            self.cluster_net.split_cluster(cluster_idx, init_weights="same")
            self.subcluster_net.split_cluster(cluster_idx, init_weights="random")

        if len(splits) > 0:
            self._update_optimizers_after_architecture_change()
            self.last_split_epoch = self.current_epoch
            logger.info(f"After split: K={self.gmm_params.k}")
            for k in range(self.gmm_params.k):
                pi_sub_0 = self.gmm_params.pi_sub[2*k].item()
                pi_sub_1 = self.gmm_params.pi_sub[2*k+1].item()
                ratio = pi_sub_0 / (pi_sub_0 + pi_sub_1) if (pi_sub_0 + pi_sub_1) > 0 else 0.5
                logger.debug(f"Cluster {k}: pi_sub=[{pi_sub_0:.4f}, {pi_sub_1:.4f}], ratio={ratio:.3f}")

        return len(splits)

    def _execute_merges(self, dataloader: DataLoader) -> int:
        """Execute merge proposals.

        Returns:
            Number of merges accepted
        """
        # Gather all data and cluster assignments
        all_data = []
        all_cluster_assignments = []

        self.cluster_net.eval()

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch

                data = data.to(self.device)

                cluster_probs = self.cluster_net(data)
                cluster_assignments = cluster_probs.argmax(dim=1)

                all_data.append(data)
                all_cluster_assignments.append(cluster_assignments)

        all_data = torch.cat(all_data, dim=0)
        all_cluster_assignments = torch.cat(all_cluster_assignments, dim=0)

        # Propose merges
        merge_config = MergeConfig(
            alpha=self.config.merge_config.alpha,
            k_nearest=self.config.merge_config.k_nearest,
            proposal_method=self.config.merge_config.proposal_method,
            stochastic_accept=self.config.merge_config.stochastic_accept
        )

        merges = propose_merges(
            self.gmm_params,
            all_data,
            all_cluster_assignments,
            self.prior,
            merge_config
        )

        for cluster_idx1, cluster_idx2 in sorted(merges, reverse=True):
            self.gmm_params.merge_clusters(
                cluster_idx1,
                cluster_idx2,
                all_data,
                self.prior
            )
            self.cluster_net.merge_clusters(cluster_idx1, cluster_idx2)
            self.subcluster_net.merge_clusters(cluster_idx1, cluster_idx2)

        if len(merges) > 0:
            self._update_optimizers_after_architecture_change()
            self.last_merge_epoch = self.current_epoch

        return len(merges)

    def _evaluate(
        self,
        dataloader: DataLoader,
        true_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate clustering performance.

        Args:
            dataloader: Data loader (not used if self.all_data is available)
            true_labels: Optional ground truth labels

        Returns:
            Dictionary with evaluation metrics
        """
        self.cluster_net.eval()

        # Use stored all_data to ensure predictions are in correct order
        # (dataloader may shuffle, causing misalignment with true_labels)
        if hasattr(self, 'all_data') and self.all_data is not None:
            with torch.no_grad():
                cluster_probs = self.cluster_net(self.all_data)
                all_predictions = cluster_probs.argmax(dim=1)
        else:
            # Fallback: gather from dataloader (may be shuffled!)
            all_predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)):
                        data = batch[0]
                    else:
                        data = batch

                    data = data.to(self.device)

                    cluster_probs = self.cluster_net(data)
                    predictions = cluster_probs.argmax(dim=1)

                    all_predictions.append(predictions)

            all_predictions = torch.cat(all_predictions, dim=0)

        metrics = {}
        if true_labels is not None:
            # Compute clustering metrics
            # Convert numpy array to tensor if needed
            if isinstance(true_labels, np.ndarray):
                true_labels = torch.from_numpy(true_labels).to(self.device)
            else:
                true_labels = true_labels.to(self.device)
            metrics = compute_clustering_metrics(all_predictions, true_labels)

        return metrics

    def _log_epoch(
        self,
        epoch: int,
        cluster_loss: float,
        subcluster_loss: float,
        num_splits: int,
        num_merges: int,
        metrics: Dict[str, float],
        epoch_time: float
    ) -> None:
        """Log epoch results."""
        logger.info(f"Epoch {epoch:3d}/{self.config.num_epochs}:")
        logger.info(f"  K: {self.gmm_params.k}")
        logger.info(f"  Cluster loss: {cluster_loss:.4f}")
        if epoch >= self.config.start_sub_clustering:
            logger.info(f"  Subcluster loss: {subcluster_loss:.4f}")
        if num_splits > 0:
            logger.info(f"  Splits: {num_splits}")
        if num_merges > 0:
            logger.info(f"  Merges: {num_merges}")
        if metrics:
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  NMI: {metrics['nmi']:.4f}")
            logger.info(f"  ARI: {metrics['ari']:.4f}")
        logger.info(f"  Time: {epoch_time:.1f}s")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'cluster_net_state_dict': self.cluster_net.state_dict(),
            'subcluster_net_state_dict': self.subcluster_net.state_dict(),
            'cluster_optimizer_state_dict': self.cluster_optimizer.state_dict(),
            'subcluster_optimizer_state_dict': self.subcluster_optimizer.state_dict(),
            'gmm_params': {
                'mus': self.gmm_params.mus,
                'covs': self.gmm_params.covs,
                'pi': self.gmm_params.pi,
                'mus_sub': self.gmm_params.mus_sub,
                'covs_sub': self.gmm_params.covs_sub,
                'pi_sub': self.gmm_params.pi_sub,
            },
            'config': self.config,
            'training_history': self.training_history,
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.cluster_net.load_state_dict(checkpoint['cluster_net_state_dict'])
        self.subcluster_net.load_state_dict(checkpoint['subcluster_net_state_dict'])
        self.cluster_optimizer.load_state_dict(checkpoint['cluster_optimizer_state_dict'])
        self.subcluster_optimizer.load_state_dict(checkpoint['subcluster_optimizer_state_dict'])

        # Restore GMM parameters
        gmm_dict = checkpoint['gmm_params']
        self.gmm_params = GMMParameters(
            mus=gmm_dict['mus'].to(self.device),
            covs=gmm_dict['covs'].to(self.device),
            pi=gmm_dict['pi'].to(self.device),
            mus_sub=gmm_dict['mus_sub'].to(self.device),
            covs_sub=gmm_dict['covs_sub'].to(self.device),
            pi_sub=gmm_dict['pi_sub'].to(self.device),
        )

        self.training_history = checkpoint['training_history']

        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
