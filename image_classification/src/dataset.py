"""データセットクラス"""
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import albumentations as A


class CIFAR10Dataset(Dataset):
    """CIFAR10用のカスタムデータセット"""
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[A.Compose] = None,
        return_index: bool = False,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            images: 画像データ (N, H, W, C) or (N, C, H, W)
            labels: ラベルデータ (N,)
            transform: Albumentations変換
            return_index: インデックスも返すかどうか
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.return_index = return_index
        if not class_names:
            raise ValueError("class_names must be provided.")
        self.class_names = list(class_names)
        
        # (N, C, H, W) -> (N, H, W, C)に変換
        if len(self.images.shape) == 4 and self.images.shape[1] == 3:
            self.images = self.images.transpose(0, 2, 3, 1)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple:
        image = self.images[idx].copy()
        label = int(self.labels[idx])
        
        # uint8に変換（Albumentations用）
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        if self.return_index:
            return image, label, idx
        
        return image, label
    
    def get_class_names(self) -> List[str]:
        return self.class_names.copy()

    def get_num_classes(self) -> int:
        return len(self.class_names)
    
    def get_labels(self) -> np.ndarray:
        """全ラベルを取得"""
        return self.labels.copy()
    
    def get_class_distribution(self) -> dict:
        """クラス分布を取得"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {
            self.class_names[int(k)]: int(v)
            for k, v in zip(unique, counts)
        }
