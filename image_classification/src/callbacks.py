"""Custom Callbacks"""
import lightning as L
from lightning.pytorch.callbacks import Callback
import torch
import numpy as np
import wandb
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json

from .visualization import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_per_class_metrics,
    plot_learning_curves,
    plot_gradcam_grid,
    plot_sample_images,
    save_classification_report,
)
from .utils import denormalize, format_time
from .dataset import CIFAR10Dataset


class TimingCallback(Callback):
    """学習・推論時間を計測するCallback"""
    
    def __init__(self):
        super().__init__()
        self.train_start_time: Optional[float] = None
        self.test_start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.epoch_times: List[float] = []
    
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.train_start_time = time.time()
        print("\n⏱️ Training started...")
    
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
            if wandb.run is not None:
                wandb.log({"timing/epoch_time_seconds": epoch_time})
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.train_start_time:
            train_time = time.time() - self.train_start_time
            avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
            
            print(f"\n{'='*60}")
            print("⏱️ Training Time Summary")
            print(f"{'='*60}")
            print(f"  Total training time:    {format_time(train_time)}")
            print(f"  Average epoch time:     {format_time(avg_epoch_time)}")
            print(f"  Total epochs:           {len(self.epoch_times)}")
            print(f"{'='*60}")
            
            if wandb.run is not None:
                wandb.log({
                    "timing/total_train_time_seconds": train_time,
                    "timing/avg_epoch_time_seconds": avg_epoch_time,
                })
    
    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.test_start_time = time.time()
        print("\n⏱️ Testing started...")
    
    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.test_start_time:
            test_time = time.time() - self.test_start_time
            num_samples = len(pl_module.all_predictions)
            throughput = num_samples / test_time
            avg_time_per_sample = test_time / num_samples * 1000  # ms
            
            print(f"\n{'='*60}")
            print("⏱️ Inference Time Summary")
            print(f"{'='*60}")
            print(f"  Total test time:        {format_time(test_time)}")
            print(f"  Number of samples:      {num_samples:,}")
            print(f"  Throughput:             {throughput:.1f} samples/sec")
            print(f"  Average inference time: {avg_time_per_sample:.2f} ms/sample")
            print(f"{'='*60}")
            
            if wandb.run is not None:
                wandb.log({
                    "timing/test_time_seconds": test_time,
                    "timing/throughput_samples_per_sec": throughput,
                    "timing/avg_inference_ms": avg_time_per_sample,
                })


class ClassificationReportCallback(Callback):
    """Classification ReportとConfusion Matrixを生成するCallback"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.class_names = list(cfg.data.class_names)
        self.dpi = cfg.visualization.dpi
    
    def on_test_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """テスト終了後にレポートを生成"""
        print("\n📊 Generating classification report and confusion matrix...")
        
        output_dir = Path(self.cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 予測と正解を取得
        predictions = pl_module.all_predictions
        targets = pl_module.all_targets
        
        # 追加情報
        additional_info = {
            "Model": self.cfg.model.name,
            "Pretrained": self.cfg.model.pretrained,
            "Epochs": self.cfg.training.epochs,
            "Batch Size": self.cfg.data.batch_size,
            "Learning Rate": self.cfg.optimizer.lr,
            "Optimizer": self.cfg.optimizer.name,
            "Scheduler": self.cfg.scheduler.name,
        }
        
        # Classification Report
        report_path = output_dir / "classification_report.txt"
        report = save_classification_report(
            targets,
            predictions,
            self.class_names,
            report_path,
            additional_info=additional_info,
        )
        print(f"\n{report}")
        
        # Confusion Matrix (通常版)
        cm_path = output_dir / "confusion_matrix.png"
        cm_fig = plot_confusion_matrix(
            targets,
            predictions,
            self.class_names,
            save_path=cm_path,
            normalize=False,
            dpi=self.dpi,
        )
        plt.close(cm_fig)
        
        # Confusion Matrix (正規化版)
        cm_norm_path = output_dir / "confusion_matrix_normalized.png"
        cm_norm_fig = plot_confusion_matrix(
            targets,
            predictions,
            self.class_names,
            save_path=cm_norm_path,
            normalize=True,
            dpi=self.dpi,
        )
        plt.close(cm_norm_fig)
        
        # W&Bにログ
        if wandb.run is not None:
            # Classification Report (.txt) as Artifact
            artifact = wandb.Artifact(
                name="classification_report",
                type="evaluation",
            )
            artifact.add_file(str(report_path))
            wandb.log_artifact(artifact)
            
            # Confusion Matrix
            wandb.log({
                "confusion_matrix/raw": wandb.Image(
                    str(cm_path),
                    caption="Confusion Matrix"
                ),
                "confusion_matrix/normalized": wandb.Image(
                    str(cm_norm_path),
                    caption="Normalized Confusion Matrix"
                ),
            })
            
            # テキストとしてもログ
            wandb.log({
                "classification_report_text": wandb.Html(f"<pre>{report}</pre>"),
            })
        
        print("✅ Classification report and confusion matrix saved!")


class PerClassMetricsCallback(Callback):
    """クラスごとのメトリクスを可視化するCallback"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.class_names = list(cfg.data.class_names)
        self.dpi = cfg.visualization.dpi
    
    def on_test_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """テスト終了後にPer-classメトリクスを可視化"""
        if not self.cfg.visualization.plot_per_class_metrics:
            return
        
        print("\n📈 Generating per-class metrics visualizations...")
        
        output_dir = Path(self.cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        predictions = pl_module.all_predictions
        targets = pl_module.all_targets
        
        # Per-class Accuracy
        acc_path = output_dir / "per_class_accuracy.png"
        acc_fig = plot_per_class_accuracy(
            targets,
            predictions,
            self.class_names,
            save_path=acc_path,
            dpi=self.dpi,
        )
        plt.close(acc_fig)
        
        # Per-class Metrics (Precision, Recall, F1)
        metrics_path = output_dir / "per_class_metrics.png"
        metrics_fig = plot_per_class_metrics(
            targets,
            predictions,
            self.class_names,
            save_path=metrics_path,
            dpi=self.dpi,
        )
        plt.close(metrics_fig)
        
        # Per-class metrics をJSONでも保存
        per_class_data = {
            "class_names": self.class_names,
            "accuracy": pl_module.per_class_acc.tolist(),
            "f1": pl_module.per_class_f1.tolist(),
            "precision": pl_module.per_class_precision.tolist(),
            "recall": pl_module.per_class_recall.tolist(),
        }
        
        with open(output_dir / "per_class_metrics.json", "w") as f:
            json.dump(per_class_data, f, indent=2)
        
        # W&Bにログ
        if wandb.run is not None:
            wandb.log({
                "per_class/accuracy": wandb.Image(
                    str(acc_path),
                    caption="Per-Class Accuracy"
                ),
                "per_class/metrics": wandb.Image(
                    str(metrics_path),
                    caption="Per-Class Metrics (Precision, Recall, F1)"
                ),
            })
            
            # テーブルとしてもログ
            table = wandb.Table(
                columns=["Class", "Accuracy", "Precision", "Recall", "F1"],
                data=[
                    [
                        self.class_names[i],
                        pl_module.per_class_acc[i],
                        pl_module.per_class_precision[i],
                        pl_module.per_class_recall[i],
                        pl_module.per_class_f1[i],
                    ]
                    for i in range(len(self.class_names))
                ],
            )
            wandb.log({"per_class/table": table})
        
        print("✅ Per-class metrics visualizations saved!")


class SampleVisualizationCallback(Callback):
    """サンプル画像を可視化するCallback"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.class_names = list(cfg.data.class_names)
        self.num_samples = cfg.visualization.num_sample_images
        self.dpi = cfg.visualization.dpi
    
    def on_test_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """テスト終了後にサンプル画像を可視化"""
        if not self.cfg.visualization.save_sample_images:
            return
        
        print("\n🖼️ Generating sample image visualizations...")
        
        output_dir = Path(self.cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_images = pl_module.all_images
        all_predictions = pl_module.all_predictions
        all_targets = pl_module.all_targets
        
        # 正解画像をサンプリング
        correct_mask = all_predictions == all_targets
        correct_indices = np.where(correct_mask)[0]
        
        if len(correct_indices) > 0:
            sample_indices = np.random.choice(
                correct_indices,
                min(self.num_samples, len(correct_indices)),
                replace=False,
            )
            
            sample_images = [denormalize(all_images[i]) for i in sample_indices]
            sample_labels = [self.class_names[all_targets[i]] for i in sample_indices]
            sample_preds = [self.class_names[all_predictions[i]] for i in sample_indices]
            
            correct_path = output_dir / "sample_correct.png"
            fig = plot_sample_images(
                sample_images,
                sample_labels,
                sample_preds,
                grid_size=4,
                title="Correctly Classified Samples",
                save_path=correct_path,
                dpi=self.dpi,
            )
            plt.close(fig)
            
            if wandb.run is not None:
                wandb.log({
                    "samples/correct": wandb.Image(
                        str(correct_path),
                        caption="Correctly Classified Samples"
                    ),
                })
        
        # 不正解画像をサンプリング
        wrong_indices = np.where(~correct_mask)[0]
        
        if len(wrong_indices) > 0:
            sample_indices = np.random.choice(
                wrong_indices,
                min(self.num_samples, len(wrong_indices)),
                replace=False,
            )
            
            sample_images = [denormalize(all_images[i]) for i in sample_indices]
            sample_labels = [self.class_names[all_targets[i]] for i in sample_indices]
            sample_preds = [self.class_names[all_predictions[i]] for i in sample_indices]
            
            wrong_path = output_dir / "sample_wrong.png"
            fig = plot_sample_images(
                sample_images,
                sample_labels,
                sample_preds,
                grid_size=4,
                title="Misclassified Samples",
                save_path=wrong_path,
                dpi=self.dpi,
            )
            plt.close(fig)
            
            if wandb.run is not None:
                wandb.log({
                    "samples/wrong": wandb.Image(
                        str(wrong_path),
                        caption="Misclassified Samples"
                    ),
                })
        
        print("✅ Sample image visualizations saved!")


class GradCAMCallback(Callback):
    """Grad-CAMの可視化を行うCallback"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.grid_size = cfg.visualization.gradcam_grid_size
        self.num_samples = cfg.visualization.num_gradcam_samples
        self.class_names = list(cfg.data.class_names)
        self.dpi = cfg.visualization.dpi

    def _select_target_layer(
        self, pl_module: L.LightningModule, input_size: tuple[int, int]
    ):
        """入力サイズに応じてGrad-CAMのターゲット層を選択"""
        model = pl_module.model
        min_size = min(input_size)

        # ResNet系: 小さな入力では深い層だと特徴マップが1x1になりやすい
        if hasattr(model, "layer2") and hasattr(model, "layer3") and hasattr(model, "layer4"):
            if min_size <= 64:
                return model.layer2[-1]
            if min_size <= 128:
                return model.layer3[-1]
            return model.layer4[-1]

        # それ以外は既存ロジックに委譲
        return pl_module.get_target_layer()
    
    def on_test_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """テスト終了後にGrad-CAMを実行"""
        print("\n🔥 Generating Grad-CAM visualizations...")
        
        output_dir = Path(self.cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # モデルを評価モードに（Grad-CAMは勾配が必要なのでno_gradは使わない）
        pl_module.eval()
        device = pl_module.device

        # 全画像と予測を取得
        all_images = pl_module.all_images
        all_predictions = pl_module.all_predictions
        all_targets = pl_module.all_targets
        all_probs = pl_module.all_probs
        
        # Grad-CAM設定
        try:
            input_h = int(all_images.shape[-2])
            input_w = int(all_images.shape[-1])
            target_layer = self._select_target_layer(pl_module, (input_h, input_w))
            target_layers = [target_layer]
        except Exception as e:
            print(f"⚠️ Could not get target layer: {e}")
            print("   Skipping Grad-CAM visualization.")
            return
        
        cam = GradCAM(
            model=pl_module.model,
            target_layers=target_layers,
        )
        
        # ランダムサンプリング
        indices = np.random.choice(
            len(all_images),
            min(self.num_samples, len(all_images)),
            replace=False,
        )
        
        random_images = []
        random_cam_images = []
        random_labels = []
        random_preds = []
        random_confs = []
        
        with torch.enable_grad():
            for idx in tqdm(indices, desc="Processing random samples"):
                img_tensor = all_images[idx].unsqueeze(0).to(device)
                img_tensor.requires_grad_(True)
                target = all_targets[idx]
                pred = all_predictions[idx]
                conf = all_probs[idx, pred]
                
                # Grad-CAM生成
                targets = [ClassifierOutputTarget(int(pred))]
                grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                # 画像を正規化解除
                img_np = denormalize(all_images[idx]).astype(np.float32)
                
                # CAMを画像に重ねる
                cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                
                random_images.append(img_np)
                random_cam_images.append(cam_image)
                random_labels.append(self.class_names[target])
                random_preds.append(self.class_names[pred])
                random_confs.append(conf)
        
        # ランダム画像のプロット
        random_fig = plot_gradcam_grid(
            random_images,
            random_cam_images,
            random_labels,
            random_preds,
            confidences=random_confs,
            grid_size=self.grid_size,
            title="Grad-CAM: Random Samples",
            save_path=output_dir / "gradcam_random.png",
            dpi=self.dpi,
        )
        plt.close(random_fig)
        
        # 間違えた画像を取得
        wrong_indices = np.where(all_predictions != all_targets)[0]
        if len(wrong_indices) > 0:
            wrong_sample_indices = np.random.choice(
                wrong_indices,
                min(self.num_samples, len(wrong_indices)),
                replace=False,
            )
            
            wrong_images = []
            wrong_cam_images = []
            wrong_labels = []
            wrong_preds = []
            wrong_confs = []
            
            with torch.enable_grad():
                for idx in tqdm(wrong_sample_indices, desc="Processing wrong predictions"):
                    img_tensor = all_images[idx].unsqueeze(0).to(device)
                    img_tensor.requires_grad_(True)
                    target = all_targets[idx]
                    pred = all_predictions[idx]
                    conf = all_probs[idx, pred]
                    
                    # Grad-CAM生成
                    targets = [ClassifierOutputTarget(int(pred))]
                    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                    
                    # 画像を正規化解除
                    img_np = denormalize(all_images[idx]).astype(np.float32)
                    
                    # CAMを画像に重ねる
                    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                    
                    wrong_images.append(img_np)
                    wrong_cam_images.append(cam_image)
                    wrong_labels.append(self.class_names[target])
                    wrong_preds.append(self.class_names[pred])
                    wrong_confs.append(conf)
            
            # 間違えた画像のプロット
            wrong_fig = plot_gradcam_grid(
                wrong_images,
                wrong_cam_images,
                wrong_labels,
                wrong_preds,
                confidences=wrong_confs,
                grid_size=self.grid_size,
                title="Grad-CAM: Wrong Predictions",
                save_path=output_dir / "gradcam_wrong.png",
                dpi=self.dpi,
            )
            plt.close(wrong_fig)
        
        # W&Bにログ
        if wandb.run is not None:
            wandb.log({
                "gradcam/random_samples": wandb.Image(
                    str(output_dir / "gradcam_random.png"),
                    caption="Grad-CAM Random Samples"
                ),
            })
            
            if len(wrong_indices) > 0:
                wandb.log({
                    "gradcam/wrong_predictions": wandb.Image(
                        str(output_dir / "gradcam_wrong.png"),
                        caption="Grad-CAM Wrong Predictions"
                    ),
                })
        
        print("✅ Grad-CAM visualizations saved!")


class LearningCurveCallback(Callback):
    """学習曲線を保存するCallback（ローカル集計）"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dpi = cfg.visualization.dpi
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
    
    @staticmethod
    def _to_float(value: Optional[torch.Tensor]) -> Optional[float]:
        if value is None:
            return None
        if hasattr(value, "item"):
            value = value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    
    def _get_metric(self, metrics: Dict[str, object], keys: List[str]) -> Optional[float]:
        for key in keys:
            if key in metrics:
                return self._to_float(metrics[key])
        return None
    
    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        
        metrics = trainer.callback_metrics
        train_loss = self._get_metric(metrics, ["train/loss_epoch", "train/loss"])
        if train_loss is not None:
            self.train_losses.append(train_loss)
        
        train_acc = self._get_metric(metrics, ["train/acc", "train/acc_epoch"])
        if train_acc is not None:
            self.train_accs.append(train_acc)
    
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        
        metrics = trainer.callback_metrics
        val_loss = self._get_metric(metrics, ["val/loss", "val/loss_epoch"])
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        val_acc = self._get_metric(metrics, ["val/acc", "val/acc_epoch"])
        if val_acc is not None:
            self.val_accs.append(val_acc)
    
    def on_train_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """学習終了後に学習曲線をプロット"""
        print("\n📈 Generating learning curves...")
        
        output_dir = Path(self.cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            train_losses = self.train_losses
            val_losses = self.val_losses
            train_accs = self.train_accs
            val_accs = self.val_accs
            
            if not train_losses or not val_losses:
                print("⚠️ Insufficient history data. Skipping learning curve plot.")
                return
            
            min_len = min(len(train_losses), len(val_losses))
            train_losses = train_losses[:min_len]
            val_losses = val_losses[:min_len]
            
            if train_accs and val_accs:
                min_acc_len = min(len(train_accs), len(val_accs), min_len)
                train_accs = train_accs[:min_acc_len]
                val_accs = val_accs[:min_acc_len]
            
            # 学習曲線をプロット
            save_path = output_dir / "learning_curves.png"
            fig = plot_learning_curves(
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                save_path=save_path,
                dpi=self.dpi,
            )
            plt.close(fig)
            
            # 履歴をJSONでも保存
            history_data = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            }
            
            with open(output_dir / "training_history.json", "w") as f:
                json.dump(history_data, f, indent=2)
            
            # W&Bにログ
            if wandb.run is not None:
                wandb.log({
                    "learning_curves": wandb.Image(
                        str(save_path),
                        caption="Learning Curves"
                    ),
                })
            
            print("✅ Learning curves saved!")
            
        except Exception as e:
            print(f"⚠️ Failed to generate learning curves: {e}")
