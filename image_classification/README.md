# 🔥 PyTorch Lightning 2.x Image Classification Boilerplate

CIFAR-10データセットを使用した画像分類のための完全なボイラープレート。

## ✨ 特徴

### 学習機能
- 🔥 **PyTorch Lightning 2.x** - クリーンな学習ループ
- 📊 **Hydra** - 設定管理
- 📈 **Weights & Biases** - 実験追跡
- ⚡ **Mixed Precision Training** - 高速化
- 🛡️ **Early Stopping** - 過学習防止
- 💾 **Model Checkpointing** - 最良モデル保存
- 🎲 **Seed固定** - 再現性確保

### 評価機能
- ✅ **Top-1/Top-5 Accuracy** - 精度評価
- 📊 **Classification Report** - 詳細なメトリクス
- 🔢 **Confusion Matrix** - 分類結果可視化
- 📈 **Per-Class Metrics** - クラスごとの精度
- 📉 **Learning Curves** - 学習曲線
- ⏱️ **Timing Metrics** - 推論時間計測

### 可視化機能
- 🔥 **Grad-CAM** - モデルの注視領域可視化
- 🖼️ **Sample Images** - 正解/不正解サンプル表示
- 📊 **Per-Class Accuracy Bar Chart** - クラス別精度

## 📁 ディレクトリ構造
image_classification/
├── configs/
│ ├── config.yaml # メイン設定
│ ├── model/
│ │ └── efficientnet.yaml # モデル設定
│ ├── data/
│ │ └── cifar10.yaml # データ設定
│ ├── optimizer/
│ │ └── adamw.yaml # オプティマイザ設定
│ ├── scheduler/
│ │ ├── cosine.yaml # CosineAnnealingLR
│ │ └── plateau.yaml # ReduceLROnPlateau
│ └── augmentation/
│ └── basic.yaml # データ拡張設定
├── src/
│ ├── init.py
│ ├── datamodule.py # LightningDataModule
│ ├── model.py # LightningModule
│ ├── dataset.py # Datasetクラス
│ ├── callbacks.py # カスタムコールバック
│ ├── utils.py # ユーティリティ関数
│ └── visualization.py # 可視化関数
├── train.py # 学習スクリプト
├── requirements.txt
└── README.md


## 🚀 セットアップ

```bash
# リポジトリをクローン/ディレクトリ作成
mkdir image_classification && cd image_classification

# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt

# W&Bログイン
wandb login

💻 使い方
基本的な学習
python train.py

設定のオーバーライド
# エポック数を変更
python train.py training.epochs=10

# バッチサイズを変更
python train.py data.batch_size=128

# 学習率を変更
python train.py optimizer.lr=5e-5

# スケジューラーを変更（Plateau）
python train.py scheduler=plateau

# 複数の設定を同時に変更
python train.py training.epochs=10 data.batch_size=128 optimizer.lr=5e-5

マルチラン（ハイパーパラメータ探索）

# 複数の学習率で実験
python train.py --multirun optimizer.lr=1e-4,5e-5,1e-5

# グリッドサーチ
python train.py --multirun optimizer.lr=1e-4,5e-5 data.batch_size=32,64,128

オフラインモード（W&Bなし）
python train.py wandb.offline=true


---

## 🎯 実行確認

```bash
# 1. ディレクトリを作成してファイルを配置
mkdir -p image_classification/{configs/{model,data,optimizer,scheduler,augmentation},src}
cd image_classification

# 2. 各ファイルを作成（上記のコードをコピー）

# 3. 依存関係をインストール
pip install -r requirements.txt

# 4. W&Bにログイン
wandb login

# 5. 学習実行
python train.py

# 6. スケジューラーを変更して実行
python train.py scheduler=plateau

# 7. エポック数を増やして実行
python train.py training.epochs=10
