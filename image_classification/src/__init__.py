"""Image Classification Package"""
from .datamodule import CIFAR10DataModule
from .model import ImageClassificationModel
from .dataset import CIFAR10Dataset
from .utils import set_seed, get_transforms, get_device_info
from .callbacks import (
    GradCAMCallback,
    ClassificationReportCallback,
    TimingCallback,
    PerClassMetricsCallback,
    SampleVisualizationCallback,
)
from .visualization import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_per_class_metrics,
    plot_learning_curves,
    plot_gradcam_grid,
    plot_sample_images,
)

__all__ = [
    # DataModule
    "CIFAR10DataModule",
    # Model
    "ImageClassificationModel",
    # Dataset
    "CIFAR10Dataset",
    # Utils
    "set_seed",
    "get_transforms",
    "get_device_info",
    # Callbacks
    "GradCAMCallback",
    "ClassificationReportCallback",
    "TimingCallback",
    "PerClassMetricsCallback",
    "SampleVisualizationCallback",
    # Visualization
    "plot_confusion_matrix",
    "plot_per_class_accuracy",
    "plot_per_class_metrics",
    "plot_learning_curves",
    "plot_gradcam_grid",
    "plot_sample_images",
]