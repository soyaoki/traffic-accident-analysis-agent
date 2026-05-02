# AI Data Analytics Agents - 交通事故統計分析

警察庁の交通事故統計（2020年・2024年）をマルチエージェントで分析するデモプロジェクトです。

## 構成

- **Layer 1: DataQueryAgent**: `pandas` を用いてデータの集計・抽出を行う低レイヤーエージェント。
- **Layer 2: TrafficSafetyAnalyst**: Layer 1 から得られたデータを、人間が作成した「分析カタログ（Layer 2）」の文脈に基づいて解釈・統合する高レイヤーエージェント。

## セットアップ

1. **環境変数の設定**
   `.env.example` を `.env` にコピーし、APIキー（OpenAI または Gemini）を設定してください。
   ```bash
   cp .env.example .env
   ```

2. **依存関係のインストール**
   `uv` を使用してインストールします。
   ```bash
   uv sync
   ```

## 実行方法

### Streamlit UI (推奨)
ブラウザで対話的に分析を行えます。
```bash
uv run streamlit run app.py
```

### CLI デモ
事前に定義されたクエリを実行します。
```bash
uv run python run.py
```

## データについて
`data/` ディレクトリに、2020年と2024年の交通事故本票データ（CSV）が配置されていることを前提としています。
これらのデータは [警察庁のオープンデータセット](https://www.npa.go.jp/publications/statistics/koutsuu/opendata/index.html) に基づいています。
