# AIICO コーホート AI エージェント

位置情報の行動データと LLM を組み合わせ、ブランド商品の **エリア別マーケティング戦略** を自動生成するシステムです。
Databricks 上で動作し、顧客の訪問履歴（コーホート行列）・属性データ・回遊行動データを統合したうえで、地点ごとの具体的な販売戦略レポートを出力します。

---

## 概要

特定ブランドの商品キーワードを入力すると、以下の流れで分析が自動実行されます。

```
ブランド名 / 商品キーワード
        │
        ▼
① LLM による商品分析レポート生成
        │
        ▼
② コーホート行列から対象エリアの訪問者 ADID を抽出
        │
        ▼
③ Embedding（BAAI/bge-m3）でLPの意味ベクトルを生成し、ADIDをスコアリング
        │
        ▼
④ 顧客属性（年齢・性別）+ 回遊行動データを LLM に渡してコンサルレポートを生成
        │
        ▼
出力：地点ごとのマーケティング戦略レポート（Parquet / CSV）
```

---

## ファイル構成

```
.
├── main_cohort.ipynb              # メインパイプライン（ADIDスコアリング・ターゲティング）
├── create_plan.ipynb              # 地点ごとのLLMコンサルティングレポート生成
├── concat_data.ipynb              # 出力ファイルの結合・CSV化
├── cohort_caption_matrix.npz      # コーホートキャプション × スポットの文脈行列
├── ブランドLP/
│   ├── 商品分析.md                 # 商品分析プロンプトテンプレート
│   └── 〇〇/                      # ブランドごとの分析結果（自動生成）
├── 生成ファイル/                     # 出力ファイル（Parquet / CSV）
├── .env.example                   # 環境変数テンプレート（要リネーム）
└── llm_agent.py                   # LLM 非同期クライアントラッパー
```

---

## セットアップ

### 前提条件

- Databricks Runtime（Delta Lake 対応）
- Python 3.12+
- Azure AI Foundry（OpenAI 互換 API）へのアクセス権

### 環境変数の設定

`.env.example` をコピーして `.env` を作成し、各値を設定してください。

```bash
cp .env.example .env
```

> **注意**: `.env` は `.gitignore` に含まれています。クレデンシャルをリポジトリにコミットしないでください。

### 依存ライブラリのインストール

各 Notebook の先頭セルで `%pip install` が実行されます。Databricks 環境では Notebook を実行することで自動的にインストールされます。

主な依存ライブラリ:

| ライブラリ | 用途 |
|---|---|
| `openai` | Azure AI Foundry への非同期 API 呼び出し |
| `sentence-transformers` | Embedding モデル（BAAI/bge-m3） |
| `delta` | Delta Lake との連携 |
| `polars` | 高速データフレーム処理 |
| `scipy` / `numpy` | 疎行列演算・L2 正規化 |
| `httpx` | 非同期 HTTP クライアント |

---

## 使い方

### 1. 商品分析レポートの生成（`main_cohort.ipynb`）

Databricks のウィジェットで以下のパラメータを設定して実行します。

| パラメータ | 説明 | 例 |
|---|---|---|
| `BRAND_NAME` | ブランド名 | `遊戯王` |
| `TARGET_BRAND_WORDS` | 商品の特徴キーワード（リスト or 文字列） | `['トレーディングカード', 'デュエル', 'コレクター']` |

実行すると以下が出力されます:
- `ブランドLP/{BRAND_NAME}.md` — LLM による商品分析レポート
- `{BRAND_NAME}_all.parquet` — スコア付き ADID リスト（全件）
- `{BRAND_NAME}_separate.parquet` — 地点ごとのカウント集計
- `{BRAND_NAME}_visible.png` — スコア分布ヒストグラム

### 2. エリア別コンサルレポートの生成（`create_plan.ipynb`）

`main_cohort.ipynb` の実行後、同じパラメータで実行します。

地点ごとに以下の観点を含む戦略レポートが生成されます:
- 顧客ペルソナとカスタマージャーニー
- プロダクト・ロケーション・フィット（商材と場所の親和性）
- 具体的なMD・送客・広告戦略
- KPI 定義

出力: `{BRAND_NAME}_analysis.parquet`

### 3. 出力ファイルの結合（`concat_data.ipynb`）

上記 2 つの出力を結合し、ポリゴンリストの住所情報を付与した最終 CSV を生成します。

```
生成ファイル/aiico_cohort_{BRAND_NAME}.csv
```

---

## アーキテクチャ

### コーホート行列（`cohort.npz`）

```
cohort.npz
├── data              : CSR 疎行列の非ゼロ値
├── indices           : 列インデックス
├── indptr            : 行スライスインデックス
├── shape             : 行列サイズ（ADID数 × コーホートキャプション数）
├── adid_list         : ADID のリスト
└── business_codelist : コーホートキャプション ID のリスト
```

各 ADID の行ベクトルは L2 正規化済みで、**確率的な行動分布**として解釈されます。

### スコアリング方式

```
LPキーワード（正/負）
    → L1正規化された重み付きベクトル
    → Embedding（bge-m3）で高次元空間に写像
    → スポット文脈行列との内積でスポット係数を算出
    → コーホート行列との内積で ADID ごとのスコアを算出
```

---

## 技術スタック

| カテゴリ | 技術 |
|---|---|
| 実行環境 | Databricks / Apache Spark |
| ストレージ | Azure Blob Storage / Delta Lake |
| LLM | Azure AI Foundry（OpenAI 互換） |
| Embedding | BAAI/bge-m3（SentenceTransformer） |
| データ処理 | PySpark / Polars / Pandas |
| 数値演算 | NumPy / SciPy（疎行列） |
| 非同期処理 | asyncio / httpx |
