"""データセットクラス"""
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import albumentations as A


class CIFAR10Dataset(Dataset):
    """CIFAR10用のカスタムデータセット"""
    
    # CIFAR-10のクラス名
    CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: Optional[A.Compose] = None,
        return_index: bool = False,
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
    
    @classmethod
    def get_class_names(cls) -> List[str]:
        return cls.CLASSES.copy()
    
    @classmethod
    def get_num_classes(cls) -> int:
        return len(cls.CLASSES)
    
    def get_labels(self) -> np.ndarray:
        """全ラベルを取得"""
        return self.labels.copy()
    
    def get_class_distribution(self) -> dict:
        """クラス分布を取得"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {
            self.CLASSES[int(k)]: int(v) 
            for k, v in zip(unique, counts)
        }