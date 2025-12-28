"""ユーティリティ関数"""
import random
import os
import sys
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import platform


def set_seed(seed: int = 42) -> None:
    """再現性のためにシードを固定"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch Lightning用
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def get_transforms(cfg: DictConfig, mode: str = "train") -> A.Compose:
    """Albumentationsの変換を構築"""
    if mode == "train":
        transform_cfg = cfg.augmentation.train
    else:
        transform_cfg = cfg.augmentation.val_test
    
    transforms = []
    for t in transform_cfg:
        transform_name = t.name
        
        # 特殊なケースの処理
        if transform_name == "ToTensorV2":
            transform_class = ToTensorV2
        elif transform_name == "ColorJitter":
            # AlbumentationsではColorJitterがないので代替
            from albumentations import ColorJitter as AColorJitter
            transform_class = AColorJitter
        else:
            transform_class = getattr(A, transform_name, None)
        
        if transform_class is None:
            print(f"⚠️ Warning: Unknown transform '{transform_name}', skipping...")
            continue
        
        params = dict(t.params) if t.params else {}
        transforms.append(transform_class(**params))
    
    return A.Compose(transforms)


def get_device_info() -> Dict[str, Any]:
    """デバイス情報を取得"""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
        ]
        info["gpu_memory"] = [
            f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB"
            for i in range(torch.cuda.device_count())
        ]
    
    return info


def detect_runtime() -> str:
    """実行環境を判定（local/colab/kaggle）"""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"):
        return "kaggle"
    if os.environ.get("COLAB_GPU") or "google.colab" in sys.modules:
        return "colab"
    return "local"


def resolve_project_root() -> Path:
    """プロジェクトルートを推定"""
    return Path(__file__).resolve().parents[2]


def resolve_data_dir(cfg: DictConfig, project_root: Path) -> Path:
    """データディレクトリを解決（相対パスはプロジェクトルート基準）"""
    env_dir = os.environ.get("IC_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    data_dir = Path(cfg.paths.data_dir)
    if data_dir.is_absolute():
        return data_dir
    return (project_root / data_dir).resolve()


def resolve_wandb_mode(cfg: DictConfig, runtime: str) -> tuple[bool, str]:
    """W&Bの有効/モードを決定"""
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False, "disabled"
    env_mode = os.environ.get("WANDB_MODE")
    if env_mode:
        return True, env_mode
    if cfg.wandb.offline:
        return True, "offline"
    if runtime == "kaggle" and not os.environ.get("WANDB_API_KEY"):
        return True, "offline"
    if os.environ.get("WANDB_API_KEY"):
        return True, "online"
    return True, "offline"


def denormalize(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """正規化を元に戻す"""
    mean = np.array(mean)
    std = np.array(std)
    
    if tensor.dim() == 4:
        # バッチの場合 (B, C, H, W) -> (B, H, W, C)
        images = tensor.cpu().numpy().transpose(0, 2, 3, 1)
        images = images * std + mean
    else:
        # 単一画像の場合 (C, H, W) -> (H, W, C)
        images = tensor.cpu().numpy().transpose(1, 2, 0)
        images = images * std + mean
    
    return np.clip(images, 0, 1)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """モデルのパラメータ数をカウント"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def format_time(seconds: float) -> str:
    """秒を読みやすい形式に変換"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def save_config(cfg: DictConfig, save_path: Path) -> None:
    """設定をファイルに保存"""
    with open(save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def get_class_weights(
    labels: np.ndarray, 
    num_classes: int,
    method: str = "inverse"
) -> torch.Tensor:
    """クラスの重みを計算（不均衡データ用）"""
    class_counts = np.bincount(labels, minlength=num_classes)
    
    if method == "inverse":
        weights = 1.0 / (class_counts + 1e-6)
    elif method == "sqrt_inverse":
        weights = 1.0 / np.sqrt(class_counts + 1e-6)
    elif method == "effective":
        # Effective Number of Samples
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
    else:
        weights = np.ones(num_classes)
    
    # 正規化
    weights = weights / weights.sum() * num_classes
    
    return torch.FloatTensor(weights)
