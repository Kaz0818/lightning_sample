"""メイン学習スクリプト"""

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.nodes import AnyNode
from omegaconf.base import ContainerMetadata, Metadata
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
import wandb
from pathlib import Path
import collections
import torch
import warnings
import typing

from src import (
    CIFAR10DataModule,
    ImageClassificationModel,
    set_seed,
    get_device_info,
    GradCAMCallback,
    ClassificationReportCallback,
    TimingCallback,
    PerClassMetricsCallback,
    SampleVisualizationCallback,
)
from src.callbacks import LearningCurveCallback
from src.utils import (
    count_parameters,
    detect_runtime,
    resolve_project_root,
    resolve_data_dir,
    resolve_wandb_mode,
)

# PyTorch 2.6以降のweights_only問題に対応
# チェックポイントにOmegaConfの設定が保存されているため、safe globalsに追加
torch.serialization.add_safe_globals(
    [
        DictConfig,
        ListConfig,
        ContainerMetadata,
        Metadata,
        typing.Any,
        dict,
        list,
        set,
        tuple,
        int,
        collections.defaultdict,
        AnyNode,
    ]
)

# 警告を抑制
warnings.filterwarnings("ignore", category=UserWarning)


def print_config(cfg: DictConfig) -> None:
    """設定を表示"""
    print("\n" + "=" * 70)
    print("📋 CONFIGURATION")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70 + "\n")


def print_device_info() -> None:
    """デバイス情報を表示"""
    info = get_device_info()

    print("\n" + "=" * 70)
    print("💻 SYSTEM INFORMATION")
    print("=" * 70)
    print(f"  Platform:        {info['platform']}")
    print(f"  Python:          {info['python_version']}")
    print(f"  PyTorch:         {info['torch_version']}")
    print(f"  CUDA Available:  {info['cuda_available']}")

    if info["cuda_available"]:
        print(f"  CUDA Version:    {info['cuda_version']}")
        print(f"  cuDNN Version:   {info['cudnn_version']}")
        print(f"  GPU Count:       {info['gpu_count']}")
        for i, (name, mem) in enumerate(zip(info["gpu_names"], info["gpu_memory"])):
            print(f"  GPU {i}:           {name} ({mem})")

    print("=" * 70 + "\n")


def print_model_info(model: L.LightningModule, cfg: DictConfig) -> None:
    """モデル情報を表示"""
    params = count_parameters(model)

    print("\n" + "=" * 70)
    print("🏗️ MODEL INFORMATION")
    print("=" * 70)
    print(f"  Model:              {cfg.model.name}")
    print(f"  Pretrained:         {cfg.model.pretrained}")
    print(f"  Num Classes:        {cfg.model.num_classes}")
    print(f"  Drop Rate:          {cfg.model.drop_rate}")
    print(f"  Total Parameters:   {params['total']:,}")
    print(f"  Trainable Params:   {params['trainable']:,}")
    print(f"  Non-trainable:      {params['non_trainable']:,}")
    print("=" * 70 + "\n")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """メイン関数"""

    # 設定を表示
    print_config(cfg)

    # デバイス情報を表示
    print_device_info()

    # シード固定
    set_seed(cfg.experiment.seed)
    L.seed_everything(cfg.experiment.seed, workers=True)

    # 実行環境の判定
    runtime = detect_runtime()

    # パス解決（Hydraのchdir影響を回避）
    project_root = resolve_project_root()
    data_dir = resolve_data_dir(cfg, project_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.data_dir = str(data_dir)

    # 出力ディレクトリ設定（Hydraの出力ディレクトリを使用）
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    cfg.paths.output_dir = str(output_dir)

    print(f"📁 Output directory: {output_dir}\n")

    # W&B初期化
    wandb_enabled, wandb_mode = resolve_wandb_mode(cfg, runtime)
    if wandb_enabled and wandb_mode != "disabled":
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment.name,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=str(output_dir),
            mode=wandb_mode,
        )
        print(f"🧪 W&B mode: {wandb_mode}")
    else:
        wandb_logger = None
        print("🧪 W&B disabled")

    # DataModule
    print("📦 Setting up DataModule...")
    datamodule = CIFAR10DataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    # Model
    print("🏗️ Building model...")
    model = ImageClassificationModel(cfg)
    print_model_info(model, cfg)

    # Callbacks
    callbacks = [
        # Checkpoints
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch:02d}-{val/acc:.4f}",
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            verbose=True,
        ),
        # Learning Rate Monitor
        LearningRateMonitor(logging_interval="step"),
        # Progress Bar
        RichProgressBar(),
        # Model Summary
        RichModelSummary(max_depth=2),
        # Custom Callbacks
        TimingCallback(),
        LearningCurveCallback(cfg),
        ClassificationReportCallback(cfg),
        PerClassMetricsCallback(cfg),
        SampleVisualizationCallback(cfg),
        GradCAMCallback(cfg),
    ]

    # Early Stopping
    if cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.early_stopping.monitor,
                patience=cfg.early_stopping.patience,
                mode=cfg.early_stopping.mode,
                min_delta=cfg.early_stopping.min_delta,
                verbose=True,
            )
        )

    # Trainer
    print("🚀 Initializing Trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices="auto",
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        deterministic=cfg.training.deterministic,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        fast_dev_run=cfg.training.fast_dev_run,
        inference_mode=False,
    )

    # デバッグ情報をW&Bにログ
    if wandb.run is not None:
        wandb.log(
            {
                "system/device_info": get_device_info(),
                "system/model_params": count_parameters(model),
                "data/stats": datamodule.get_data_stats(),
            }
        )

    # 学習
    print("\n" + "=" * 70)
    print("🎯 STARTING TRAINING")
    print("=" * 70)
    trainer.fit(model, datamodule=datamodule)

    # ベストモデルでテスト
    print("\n" + "=" * 70)
    print("🧪 TESTING WITH BEST MODEL")
    print("=" * 70)

    # ベストチェックポイントのパスを取得
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        trainer.test(model, datamodule=datamodule, ckpt_path=best_model_path)
    else:
        print("No best model checkpoint found. Testing with current model.")
        trainer.test(model, datamodule=datamodule)

    # 最終結果を表示
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)

    def _metric_to_scalar(value: object) -> float | None:
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else None
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    test_metrics: dict[str, float] = {}
    for key, value in trainer.callback_metrics.items():
        if not key.startswith("test/"):
            continue
        scalar_value = _metric_to_scalar(value)
        if scalar_value is not None:
            test_metrics[key.replace("test/", "")] = scalar_value

    for metric, value in test_metrics.items():
        print(f"  {metric:20s}: {value:.4f}")

    print("=" * 70)

    # W&B終了
    wandb.finish()

    # 完了メッセージ
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 All outputs saved to: {output_dir}")
    print("\n📋 Output files:")
    for file in sorted(output_dir.glob("*")):
        if file.is_file():
            print(f"    - {file.name}")

    checkpoints_dir = output_dir / "checkpoints"
    if checkpoints_dir.exists():
        print("\n📦 Checkpoints:")
        for file in sorted(checkpoints_dir.glob("*.ckpt")):
            print(f"    - {file.name}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
