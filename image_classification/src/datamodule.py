"""Lightning DataModule"""
import lightning as L
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
import numpy as np
from omegaconf import DictConfig
from typing import Optional, Dict
import torch

from .dataset import CIFAR10Dataset
from .utils import get_transforms


class CIFAR10DataModule(L.LightningDataModule):
    """CIFAR10用のLightning DataModule"""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.data
        
        self.train_dataset: Optional[CIFAR10Dataset] = None
        self.val_dataset: Optional[CIFAR10Dataset] = None
        self.test_dataset: Optional[CIFAR10Dataset] = None
        
        # Transforms
        self.train_transform = get_transforms(cfg, mode="train")
        self.val_transform = get_transforms(cfg, mode="val")
        self.test_transform = get_transforms(cfg, mode="test")
        
        # データ統計
        self.data_stats: Dict = {}
    
    def prepare_data(self) -> None:
        """データのダウンロード"""
        CIFAR10(root=self.cfg.paths.data_dir, train=True, download=True)
        CIFAR10(root=self.cfg.paths.data_dir, train=False, download=True)
    
    def setup(self, stage: Optional[str] = None) -> None:
        """データの分割とデータセットの作成"""
        # 学習データ読み込み
        train_data = CIFAR10(
            root=self.cfg.paths.data_dir,
            train=True,
            download=False,
        )
        
        # NumPy配列に変換
        train_images = np.array(train_data.data)  # (50000, 32, 32, 3)
        train_labels = np.array(train_data.targets)
        
        # Train/Val/Test分割 (80:10:10)
        # まずTrain+ValとTestに分割 (90:10)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            train_images,
            train_labels,
            test_size=self.data_cfg.test_ratio,
            random_state=self.cfg.experiment.seed,
            stratify=train_labels,
        )
        
        # 次にTrainとValに分割 (80:10 = 8:1)
        val_ratio_adjusted = self.data_cfg.val_ratio / (
            self.data_cfg.train_ratio + self.data_cfg.val_ratio
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_ratio_adjusted,
            random_state=self.cfg.experiment.seed,
            stratify=y_trainval,
        )
        
        # データ統計を保存
        self.data_stats = {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "num_classes": self.data_cfg.num_classes,
            "img_size": self.data_cfg.img_size,
            "train_class_dist": self._get_class_distribution(y_train),
            "val_class_dist": self._get_class_distribution(y_val),
            "test_class_dist": self._get_class_distribution(y_test),
        }
        
        print(f"\n{'='*60}")
        print("📊 Dataset Statistics")
        print(f"{'='*60}")
        print(f"  Train samples: {len(X_train):,}")
        print(f"  Val samples:   {len(X_val):,}")
        print(f"  Test samples:  {len(X_test):,}")
        print(f"  Total:         {len(X_train) + len(X_val) + len(X_test):,}")
        print(f"  Image size:    {self.data_cfg.img_size}x{self.data_cfg.img_size}")
        print(f"  Num classes:   {self.data_cfg.num_classes}")
        print(f"{'='*60}\n")
        
        # データセット作成
        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR10Dataset(
                images=X_train,
                labels=y_train,
                transform=self.train_transform,
            )
            self.val_dataset = CIFAR10Dataset(
                images=X_val,
                labels=y_val,
                transform=self.val_transform,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10Dataset(
                images=X_test,
                labels=y_test,
                transform=self.test_transform,
            )
    
    def _get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """クラス分布を取得"""
        class_names = CIFAR10Dataset.get_class_names()
        unique, counts = np.unique(labels, return_counts=True)
        return {class_names[int(k)]: int(v) for k, v in zip(unique, counts)}
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory,
            persistent_workers=self.data_cfg.persistent_workers if self.data_cfg.num_workers > 0 else False,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory,
            persistent_workers=self.data_cfg.persistent_workers if self.data_cfg.num_workers > 0 else False,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=self.data_cfg.pin_memory,
            persistent_workers=self.data_cfg.persistent_workers if self.data_cfg.num_workers > 0 else False,
        )
    
    @property
    def num_classes(self) -> int:
        return self.data_cfg.num_classes
    
    @property
    def class_names(self) -> list:
        return CIFAR10Dataset.get_class_names()
    
    def get_data_stats(self) -> Dict:
        return self.data_stats