"""可視化関数"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# スタイル設定
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    normalize: bool = False,
    figsize: tuple = (12, 10),
    dpi: int = 150,
) -> plt.Figure:
    """Confusion Matrixをプロット"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
    dpi: int = 150,
) -> plt.Figure:
    """クラスごとの精度をプロット"""
    # クラスごとの精度計算
    accuracies = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
        else:
            acc = 0.0
        accuracies.append(acc)
    
    # DataFrameに変換してソート
    df = pd.DataFrame({
        "Class": class_names,
        "Accuracy": accuracies,
    })
    df = df.sort_values("Accuracy", ascending=True)
    
    # プロット
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn(df["Accuracy"].values)
    bars = ax.barh(df["Class"], df["Accuracy"], color=colors, edgecolor="black", linewidth=0.5)
    
    # 値をバーの横に表示
    for bar, acc in zip(bars, df["Accuracy"]):
        ax.text(
            bar.get_width() + 0.01, 
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1%}",
            va="center",
            fontsize=10,
        )
    
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.axvline(x=np.mean(accuracies), color="red", linestyle="--", label=f"Mean: {np.mean(accuracies):.1%}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 8),
    dpi: int = 150,
) -> plt.Figure:
    """クラスごとのPrecision/Recall/F1をプロット"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # DataFrameに変換
    df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support,
    })
    
    # F1でソート
    df = df.sort_values("F1-Score", ascending=True)
    
    # プロット
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics = ["Precision", "Recall", "F1-Score"]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    
    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.barh(df["Class"], df[metric], color=color, edgecolor="black", linewidth=0.5, alpha=0.8)
        
        for bar, val in zip(bars, df[metric]):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=9,
            )
        
        ax.set_xlabel(metric, fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Class", fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.axvline(x=df[metric].mean(), color="red", linestyle="--", alpha=0.7)
    
    plt.suptitle("Per-Class Metrics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 5),
    dpi: int = 150,
) -> plt.Figure:
    """学習曲線をプロット"""
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss Plot
    if train_losses and val_losses:
        min_len = min(len(train_losses), len(val_losses))
        epochs = range(1, min_len + 1)
        axes[0].plot(epochs, train_losses[:min_len], "b-o", label="Train", markersize=4, linewidth=2)
        axes[0].plot(epochs, val_losses[:min_len], "r-s", label="Validation", markersize=4, linewidth=2)
        
        best_epoch = np.argmin(val_losses[:min_len]) + 1
        axes[0].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                        label=f"Best: Epoch {best_epoch}")
        axes[0].legend(loc="upper right")
    
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Loss Curve", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy Plot
    if train_accs and val_accs:
        min_len = min(len(train_accs), len(val_accs))
        epochs = range(1, min_len + 1)
        axes[1].plot(epochs, train_accs[:min_len], "b-o", label="Train", markersize=4, linewidth=2)
        axes[1].plot(epochs, val_accs[:min_len], "r-s", label="Validation", markersize=4, linewidth=2)
        
        best_epoch = np.argmax(val_accs[:min_len]) + 1
        axes[1].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                        label=f"Best: Epoch {best_epoch}")
        axes[1].legend(loc="lower right")
    
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Accuracy Curve", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_gradcam_grid(
    images: List[np.ndarray],
    cam_images: List[np.ndarray],
    labels: List[str],
    predictions: List[str],
    confidences: Optional[List[float]] = None,
    grid_size: int = 4,
    title: str = "Grad-CAM Visualization",
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Grad-CAMをグリッドでプロット"""
    n_images = min(len(images), grid_size * grid_size)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    for idx in range(grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col]
        
        if idx < n_images:
            # CAM画像を表示
            ax.imshow(cam_images[idx])
            
            # 正解・不正解で色を変える
            is_correct = labels[idx] == predictions[idx]
            color = "green" if is_correct else "red"
            
            # タイトル設定
            title_text = f"True: {labels[idx]}\nPred: {predictions[idx]}"
            if confidences is not None and idx < len(confidences):
                title_text += f"\nConf: {confidences[idx]:.1%}"
            
            ax.set_title(title_text, fontsize=9, color=color, fontweight="bold" if not is_correct else "normal")
        
        ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_sample_images(
    images: List[np.ndarray],
    labels: List[str],
    predictions: Optional[List[str]] = None,
    grid_size: int = 4,
    title: str = "Sample Images",
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """サンプル画像をグリッドでプロット"""
    n_images = min(len(images), grid_size * grid_size)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    
    for idx in range(grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col]
        
        if idx < n_images:
            ax.imshow(images[idx])
            
            if predictions is not None:
                is_correct = labels[idx] == predictions[idx]
                color = "green" if is_correct else "red"
                title_text = f"T: {labels[idx]}\nP: {predictions[idx]}"
            else:
                color = "black"
                title_text = labels[idx]
            
            ax.set_title(title_text, fontsize=9, color=color)
        
        ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Path,
    additional_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Classification Reportを保存"""
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
    
    # 追加統計
    total = len(y_true)
    correct = (y_true == y_pred).sum()
    accuracy = correct / total
    
    with open(save_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        if additional_info:
            f.write("Model Information:\n")
            f.write("-" * 40 + "\n")
            for key, value in additional_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        f.write("Overall Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total Samples:    {total:,}\n")
        f.write(f"  Correct:          {correct:,}\n")
        f.write(f"  Incorrect:        {total - correct:,}\n")
        f.write(f"  Overall Accuracy: {accuracy:.4f} ({accuracy:.2%})\n")
        f.write("\n")
        
        f.write("Detailed Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(report)
        f.write("\n")
        
        # クラスごとのサンプル数
        f.write("Class Distribution (in test set):\n")
        f.write("-" * 40 + "\n")
        for i, name in enumerate(class_names):
            count = (y_true == i).sum()
            f.write(f"  {name:15s}: {count:5d} ({count/total:.1%})\n")
    
    return report