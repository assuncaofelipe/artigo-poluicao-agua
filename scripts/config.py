from dataclasses import dataclass


@dataclass
class Config:
    data_path: str = "./scripts/data/dados_serpa.csv"
    feature_cols: tuple = ("x", "y", "TEMP", "COR_A", "TURB", "COR_V", "OD")
    target_col: str = "pH"
    test_split: float = 0.2
    val_split: float = 0.1
    seed: int = 42
    batch_size: int = 16
    epochs: int = 1000
    lr: float = 2e-3
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    # augmentation / validation
    augment_noise_std: float = 0.02
    augment_factor: int = 5
    k_folds: int = 5
    # grid search options
    grid_hidden_sizes: tuple = (16, 32, 64)
    grid_time_steps: tuple = (15, 25, 35)
    grid_lr: tuple = (5e-4, 1e-3, 2e-3)
    grid_beta: tuple = (0.9, 0.95)
    grid_thresh: tuple = (0.8, 1.0)
    # augmentation search
    grid_aug_std: tuple = (0.005, 0.01, 0.02, 0.03)
    time_steps: int = 15
    hidden_size: int = 64
    beta: float = 0.95
    thresh: float = 0.8
    device: str = "cpu"  # set to "cuda" if available
    out_dir: str = "./outputs"
    fig_dir: str = "./figuras"
    dpi: int = 600
