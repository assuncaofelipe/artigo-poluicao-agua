from dataclasses import dataclass


@dataclass
class Config:
    data_path: str = "../scripts/data/dados_serpa.csv"
    # Keep target variables out of the input to avoid leakage in multimodal training.
    feature_cols: tuple = ("x", "y", "TEMP", "COR_A", "COR_V")
    target_cols: tuple = ("pH", "OD", "TURB")
    test_split: float = 0.2
    val_split: float = 0.1
    seed: int = 42
    batch_size: int = 16
    epochs: int = 500
    lr: float = 2e-3
    time_steps: int = 15
    hidden_size: int = 64
    beta: float = 0.95
    thresh: float = 0.8
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    augment_noise_std: float = 0.01
    augment_factor: int = 5
    normalize_targets: bool = True
    device: str = "cpu"
    out_dir: str = "./outputs"
    fig_dir: str = "./figuras"
    dpi: int = 600
    # pollution index (weights sum to 1.0)
    pollution_weights: tuple = (0.4, 0.4, 0.2)  # (pH, OD, TURB)
    ph_target: float = 7.0
    # loss weights per target (pH, OD, TURB)
    target_loss_weights: tuple = (0.2, 0.2, 0.6)
    # grid search options
    grid_hidden_sizes: tuple = (32, 64, 96)
    grid_time_steps: tuple = (10, 15, 25)
    grid_lr: tuple = (5e-4, 1e-3, 2e-3)
    grid_beta: tuple = (0.9, 0.95)
    grid_thresh: tuple = (0.8, 1.0)
    grid_aug_std: tuple = (0.01, 0.02, 0.03)
