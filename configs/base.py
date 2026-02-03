"""Base configuration classes for DeepDPM training."""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path

# Valid values for enum-like fields
VALID_CLUSTER_LOSS_TYPES = {"isotropic", "KL_GMM_2"}
VALID_SUBCLUSTER_LOSS_TYPES = {"isotropic"}
VALID_PROPOSAL_METHODS = {"kmeans", "brute_force_NN"}
VALID_DEVICES = {"cuda", "cpu"}
VALID_LR_SCHEDULER_MODES = {"min", "max"}


@dataclass
class NIWPriorConfig:
    """Normal-Inverse-Wishart prior configuration."""
    kappa: float = 0.0001  # Prior pseudocount for mean (weak prior)
    nu_offset: int = 2  # nu = D + nu_offset
    psi_scale: float = 0.005  # Ψ = I * psi_scale


@dataclass
class SplitConfig:
    """Split proposal configuration."""
    alpha: float = 10.0  # Dirichlet process concentration
    cov_const: float = 0.005  # Covariance constant (legacy, not used)
    stochastic_accept: bool = True  # Stochastic vs deterministic acceptance
    min_cluster_size: int = 2  # Minimum points to consider for split
    ignore_subclusters: bool = False  # If True, use min-distance instead of SubclusterNet


@dataclass
class MergeConfig:
    """Merge proposal configuration."""
    alpha: float = 10.0  # Dirichlet process concentration (used in Hastings ratio)
    k_nearest: int = 3  # Number of nearest neighbors to consider
    proposal_method: str = "kmeans"  # "kmeans" or "brute_force_NN"
    stochastic_accept: bool = True  # Stochastic vs deterministic acceptance


@dataclass
class TrainingConfig:
    """Main training configuration for DeepDPM."""

    # Model architecture
    input_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [50])  # Single hidden layer
    init_k: int = 1
    cluster_softmax_norm: float = 1.0  # Temperature scaling for cluster net softmax
    subcluster_softmax_norm: float = 1.0  # Temperature scaling for subcluster net softmax

    # Training hyperparameters
    batch_size: int = 128
    num_epochs: int = 500
    cluster_lr: float = 0.0005
    subcluster_lr: float = 0.005

    # Learning rate scheduling
    use_cluster_lr_scheduler: bool = False  # Disabled by default
    use_subcluster_lr_scheduler: bool = False  # No LR scheduling for SubclusterNet
    lr_scheduler_patience: int = 4  # Epochs with no improvement before reducing LR
    lr_scheduler_factor: float = 0.5  # Factor to reduce LR by
    lr_scheduler_mode: str = "min"  # "min" for loss minimization

    # Loss configuration
    cluster_loss_type: str = "KL_GMM_2"  # "isotropic" or "KL_GMM_2"
    subcluster_loss_type: str = "isotropic"
    cluster_loss_weight: float = 1.0
    subcluster_loss_weight: float = 1.0

    # Training schedule
    gmm_warmup_epochs: int = 25  # Epochs to train ClusterNet before updating GMM
    start_sub_clustering: int = 45  # Epoch to start training subclusters
    subcluster_warmup_epochs: int = 0  # Epochs to train subcluster net before updating GMM subclusters
    start_splitting: int = 55  # Epoch to start split proposals
    start_merging: int = 55  # Epoch to start merge proposals
    split_merge_every_n_epochs: int = 30  # Frequency of split/merge checks
    eval_every_n_epochs: int = 10  # Frequency of evaluation
    freeze_mus_after_split_merge: int = 5  # Freeze GMM updates for N epochs after splits/merges
    freeze_mus_after_init: int = 5  # Freeze GMM updates for N epochs after K-means initialization

    # Prior and split/merge configs
    prior_config: NIWPriorConfig = field(default_factory=NIWPriorConfig)
    split_config: SplitConfig = field(default_factory=SplitConfig)
    merge_config: MergeConfig = field(default_factory=MergeConfig)

    # Reproducibility
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"

    # Logging and checkpointing
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 50
    verbose: bool = True

    # Memory management
    max_history_size: int = 500  # Maximum entries in training_history (0 = unlimited)

    def __post_init__(self):
        """Validate configuration on instantiation."""
        self.validate()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            TrainingConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Handle nested configs
        if 'prior' in config_dict:
            config_dict['prior_config'] = NIWPriorConfig(**config_dict.pop('prior'))

        if 'split' in config_dict:
            config_dict['split_config'] = SplitConfig(**config_dict.pop('split'))

        if 'merge' in config_dict:
            config_dict['merge_config'] = MergeConfig(**config_dict.pop('merge'))

        # Handle model and training sub-dicts
        if 'model' in config_dict:
            model_dict = config_dict.pop('model')
            config_dict.update(model_dict)

        if 'training' in config_dict:
            training_dict = config_dict.pop('training')
            config_dict.update(training_dict)

        config = cls(**config_dict)
        # Note: validate() is called automatically via __post_init__
        return config

    def validate(self):
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        errors = []

        # Type checks
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            errors.append(f"input_dim must be positive integer, got {self.input_dim}")

        if not isinstance(self.hidden_dims, list) or not all(isinstance(d, int) and d > 0 for d in self.hidden_dims):
            errors.append(f"hidden_dims must be list of positive integers, got {self.hidden_dims}")

        if not isinstance(self.init_k, int) or self.init_k <= 0:
            errors.append(f"init_k must be positive integer, got {self.init_k}")

        # Range validation for learning rates
        if self.cluster_lr <= 0:
            errors.append(f"cluster_lr must be positive, got {self.cluster_lr}")

        if self.subcluster_lr <= 0:
            errors.append(f"subcluster_lr must be positive, got {self.subcluster_lr}")

        # Range validation for batch size and epochs
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")

        if self.num_epochs <= 0:
            errors.append(f"num_epochs must be positive, got {self.num_epochs}")

        # Enum validation for loss types
        if self.cluster_loss_type not in VALID_CLUSTER_LOSS_TYPES:
            errors.append(f"cluster_loss_type must be one of {VALID_CLUSTER_LOSS_TYPES}, got {self.cluster_loss_type}")

        if self.subcluster_loss_type not in VALID_SUBCLUSTER_LOSS_TYPES:
            errors.append(f"subcluster_loss_type must be one of {VALID_SUBCLUSTER_LOSS_TYPES}, got {self.subcluster_loss_type}")

        # Enum validation for merge proposal method
        if self.merge_config.proposal_method not in VALID_PROPOSAL_METHODS:
            errors.append(f"merge_config.proposal_method must be one of {VALID_PROPOSAL_METHODS}, got {self.merge_config.proposal_method}")

        # Device validation
        if self.device not in VALID_DEVICES:
            errors.append(f"device must be one of {VALID_DEVICES}, got {self.device}")

        # LR scheduler mode validation
        if self.lr_scheduler_mode not in VALID_LR_SCHEDULER_MODES:
            errors.append(f"lr_scheduler_mode must be one of {VALID_LR_SCHEDULER_MODES}, got {self.lr_scheduler_mode}")

        # Consistency checks
        if self.start_splitting < self.gmm_warmup_epochs:
            errors.append(f"start_splitting ({self.start_splitting}) should be >= gmm_warmup_epochs ({self.gmm_warmup_epochs})")

        if self.start_merging < self.gmm_warmup_epochs:
            errors.append(f"start_merging ({self.start_merging}) should be >= gmm_warmup_epochs ({self.gmm_warmup_epochs})")

        if self.start_sub_clustering < self.gmm_warmup_epochs:
            errors.append(f"start_sub_clustering ({self.start_sub_clustering}) should be >= gmm_warmup_epochs ({self.gmm_warmup_epochs})")

        # Prior validation
        if self.prior_config.kappa <= 0:
            errors.append(f"prior_config.kappa must be positive, got {self.prior_config.kappa}")

        if self.prior_config.nu_offset < 0:
            errors.append(f"prior_config.nu_offset must be non-negative, got {self.prior_config.nu_offset}")

        if self.prior_config.psi_scale <= 0:
            errors.append(f"prior_config.psi_scale must be positive, got {self.prior_config.psi_scale}")

        # Split config validation
        if self.split_config.alpha <= 0:
            errors.append(f"split_config.alpha must be positive, got {self.split_config.alpha}")

        if self.split_config.min_cluster_size < 1:
            errors.append(f"split_config.min_cluster_size must be >= 1, got {self.split_config.min_cluster_size}")

        # Merge config validation
        if self.merge_config.alpha <= 0:
            errors.append(f"merge_config.alpha must be positive, got {self.merge_config.alpha}")

        if self.merge_config.k_nearest < 1:
            errors.append(f"merge_config.k_nearest must be >= 1, got {self.merge_config.k_nearest}")

        if errors:
            raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        config_dict = {
            'model': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'init_k': self.init_k,
                'cluster_softmax_norm': self.cluster_softmax_norm,
                'subcluster_softmax_norm': self.subcluster_softmax_norm,
            },
            'training': {
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'cluster_lr': self.cluster_lr,
                'subcluster_lr': self.subcluster_lr,
                'cluster_loss_type': self.cluster_loss_type,
                'subcluster_loss_type': self.subcluster_loss_type,
                'cluster_loss_weight': self.cluster_loss_weight,
                'subcluster_loss_weight': self.subcluster_loss_weight,
                'start_sub_clustering': self.start_sub_clustering,
                'start_splitting': self.start_splitting,
                'start_merging': self.start_merging,
                'split_merge_every_n_epochs': self.split_merge_every_n_epochs,
                'eval_every_n_epochs': self.eval_every_n_epochs,
            },
            'prior': {
                'kappa': self.prior_config.kappa,
                'nu_offset': self.prior_config.nu_offset,
                'psi_scale': self.prior_config.psi_scale,
            },
            'split': {
                'alpha': self.split_config.alpha,
                'stochastic_accept': self.split_config.stochastic_accept,
                'min_cluster_size': self.split_config.min_cluster_size,
            },
            'merge': {
                'alpha': self.merge_config.alpha,
                'k_nearest': self.merge_config.k_nearest,
                'proposal_method': self.merge_config.proposal_method,
                'stochastic_accept': self.merge_config.stochastic_accept,
            },
            'seed': self.seed,
            'device': self.device,
            'log_dir': self.log_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'save_every_n_epochs': self.save_every_n_epochs,
            'verbose': self.verbose,
        }

        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
