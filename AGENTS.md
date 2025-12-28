# Repository Guidelines

## Project Structure & Module Organization
- ルート直下は簡易エントリの `main.py` と Python プロジェクト定義の `pyproject.toml` が中心です。
- 実装の主対象は `image_classification/` 配下で、学習スクリプト `train.py`、実装本体 `src/`、Hydra 設定 `configs/` を置きます。
- 実データは `data/` と `image_classification/data/` に配置します。どちらも `.gitignore` 対象です。
- 学習成果物は `outputs/` と `image_classification/outputs/` に保存され、こちらも Git 管理外です。

## Build, Test, and Development Commands
- Python 3.12+ を前提に、依存はサブプロジェクトの requirements を使います。
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r image_classification/requirements.txt
```
- W&B を使う場合は事前にログインします。
```bash
wandb login
```
- 学習の基本実行と設定上書き例です。
```bash
python image_classification/train.py
python image_classification/train.py training.epochs=10 data.batch_size=128
python image_classification/train.py scheduler=plateau
python image_classification/train.py --multirun optimizer.lr=1e-4,5e-5
```

## Coding Style & Naming Conventions
- 4 スペースインデント、PEP 8 に近い書き方、型ヒント併用のスタイルです。
- 命名は `snake_case`（関数・変数）と `PascalCase`（クラス）を踏襲してください。
- Hydra のキーは `training.*` / `data.*` / `model.*` など既存の階層に合わせます。
- 自動フォーマッタ／リンタの設定は現状見当たらないため、既存コードの体裁に合わせてください。

## Testing Guidelines
- 現時点でテストスイートは確認できません。テスト導入時は `image_classification/tests/` などに集約し、実行コマンドを README に追記してください。
- カバレッジ目標やフレームワークの規定も未設定です。

## Commit & Pull Request Guidelines
- Git 履歴が空のため、コミットメッセージ規約は未確定です。短い要約 + 必要ならスコープ（例: `feat:`, `fix:`）で統一することを推奨します。
- PR では「目的／変更点／実行手順／結果（主要メトリクスや図）」を簡潔に記載してください。
- `data/` と `outputs/` は追加しないでください。設定変更は差分が追える粒度でまとめます。

## Security & Configuration Tips
- W&B などの API キーは環境変数で扱い、リポジトリに残さないでください。
- 大きな学習成果物やデータは Git 管理外のパスに保存します。
