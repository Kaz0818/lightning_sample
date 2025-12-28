"""Lightning Module for Image Classification"""
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import MulticlassAccuracy
from omegaconf import DictConfig
from typing import Dict, Any, Tuple, List


class ImageClassificationModel(L.LightningModule):
    """画像分類用のLightning Module"""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # モデル構築
        self.model = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes,
            drop_rate=cfg.model.drop_rate,
            drop_path_rate=cfg.model.drop_path_rate,
        )
        
        # 損失関数
        self.criterion = nn.CrossEntropyLoss()
        
        # メトリクス
        num_classes = cfg.model.num_classes
        
        # Train metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_top5_acc = MulticlassAccuracy(num_classes=num_classes, top_k=5, average="micro")
        
        # Val metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_top5_acc = MulticlassAccuracy(num_classes=num_classes, top_k=5, average="micro")
        
        # Test metrics
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_top5_acc = MulticlassAccuracy(num_classes=num_classes, top_k=5, average="micro")
        
        # Per-class metrics
        self.test_per_class_acc = MulticlassAccuracy(num_classes=num_classes, average=None)
        self.test_per_class_f1 = F1Score(task="multiclass", num_classes=num_classes, average=None)
        self.test_per_class_precision = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.test_per_class_recall = Recall(task="multiclass", num_classes=num_classes, average=None)
        
        # テスト時の予測を保存
        self.test_predictions: List[torch.Tensor] = []
        self.test_targets: List[torch.Tensor] = []
        self.test_images: List[torch.Tensor] = []
        self.test_logits: List[torch.Tensor] = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> Dict[str, torch.Tensor]:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "logits": logits,
            "probs": probs,
        }
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self._shared_step(batch, "train")
        
        # メトリクス更新
        self.train_acc(outputs["preds"], outputs["labels"])
        self.train_top5_acc(outputs["logits"], outputs["labels"])
        
        # ログ
        self.log("train/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/top5_acc", self.train_top5_acc, on_step=False, on_epoch=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)
        
        return outputs["loss"]
    
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        outputs = self._shared_step(batch, "val")
        
        # メトリクス更新
        self.val_acc(outputs["preds"], outputs["labels"])
        self.val_f1(outputs["preds"], outputs["labels"])
        self.val_top5_acc(outputs["logits"], outputs["labels"])
        
        # ログ
        self.log("val/loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/top5_acc", self.val_top5_acc, on_step=False, on_epoch=True)
    
    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        images, labels = batch
        outputs = self._shared_step(batch, "test")
        
        # メトリクス更新
        self.test_acc(outputs["preds"], outputs["labels"])
        self.test_f1(outputs["preds"], outputs["labels"])
        self.test_precision(outputs["preds"], outputs["labels"])
        self.test_recall(outputs["preds"], outputs["labels"])
        self.test_top5_acc(outputs["logits"], outputs["labels"])
        
        # Per-class metrics
        self.test_per_class_acc(outputs["preds"], outputs["labels"])
        self.test_per_class_f1(outputs["preds"], outputs["labels"])
        self.test_per_class_precision(outputs["preds"], outputs["labels"])
        self.test_per_class_recall(outputs["preds"], outputs["labels"])
        
        # 予測を保存（後でGrad-CAMやレポート用に使用）
        self.test_predictions.append(outputs["preds"].cpu())
        self.test_targets.append(outputs["labels"].cpu())
        self.test_images.append(images.cpu())
        self.test_logits.append(outputs["logits"].cpu())
        
        # ログ
        self.log("test/loss", outputs["loss"], on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test/top5_acc", self.test_top5_acc, on_step=False, on_epoch=True)
    
    def on_test_epoch_end(self) -> None:
        """テスト終了時に予測を結合"""
        self.all_predictions = torch.cat(self.test_predictions, dim=0).numpy()
        self.all_targets = torch.cat(self.test_targets, dim=0).numpy()
        self.all_images = torch.cat(self.test_images, dim=0)
        self.all_logits = torch.cat(self.test_logits, dim=0)
        self.all_probs = F.softmax(self.all_logits, dim=1).numpy()
        
        # Per-class metricsを取得
        self.per_class_acc = self.test_per_class_acc.compute().cpu().numpy()
        self.per_class_f1 = self.test_per_class_f1.compute().cpu().numpy()
        self.per_class_precision = self.test_per_class_precision.compute().cpu().numpy()
        self.per_class_recall = self.test_per_class_recall.compute().cpu().numpy()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=tuple(self.cfg.optimizer.betas),
            eps=self.cfg.optimizer.eps,
            amsgrad=self.cfg.optimizer.amsgrad,
        )
        
        # Scheduler
        if self.cfg.scheduler.name == "cosine":
            T_max = self.cfg.scheduler.T_max or self.cfg.training.epochs
            
            if self.cfg.scheduler.warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=self.cfg.scheduler.warmup_start_lr / self.cfg.optimizer.lr,
                    total_iters=self.cfg.scheduler.warmup_epochs,
                )
                main_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=T_max - self.cfg.scheduler.warmup_epochs,
                    eta_min=self.cfg.scheduler.eta_min,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[self.cfg.scheduler.warmup_epochs],
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                    eta_min=self.cfg.scheduler.eta_min,
                )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        
        elif self.cfg.scheduler.name == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.cfg.scheduler.mode,
                factor=self.cfg.scheduler.factor,
                patience=self.cfg.scheduler.patience,
                min_lr=self.cfg.scheduler.min_lr,
                threshold=self.cfg.scheduler.threshold,
                cooldown=self.cfg.scheduler.cooldown,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }
        else:
            return {"optimizer": optimizer}
    
    def get_target_layer(self) -> nn.Module:
        """Grad-CAM用のターゲット層を取得"""
        if hasattr(self.model, "conv_head"):
            return self.model.conv_head
        elif hasattr(self.model, "features"):
            return self.model.features[-1]
        else:
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    return module
            raise ValueError("Could not find target layer for Grad-CAM")