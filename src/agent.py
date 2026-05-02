"""エージェント構成（OAI Agents SDK）。
SQL (Data Retrieval) + Python (Data Analysis/Visualization) のハイブリッド構成。
"""

import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from src.tools import ALL_TOOLS
from src.preprocess import init_db

load_dotenv(override=True)
init_db()

_GEMINI_BASE_URL_STUDIO = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_PREFIXES = ("gemini-", "google/gemini-")

def _resolve_model(model_name: str) -> str | OpenAIChatCompletionsModel:
    if any(model_name.startswith(p) for p in _GEMINI_PREFIXES):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise ValueError("GEMINI_API_KEY が未設定です。")
        if model_name.startswith("google/"):
            project_id = os.getenv("VERTEX_PROJECT_ID")
            region = os.getenv("VERTEX_REGION", "us-central1")
            base_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints/openapi"
            wait_time = 0.0
        else:
            base_url = _GEMINI_BASE_URL_STUDIO
            wait_time = 1.0
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        original_create = client.chat.completions.create
        async def delayed_create(*args, **kwargs):
            if wait_time > 0: await asyncio.sleep(wait_time)
            kwargs.pop("parallel_tool_calls", None)
            return await original_create(*args, **kwargs)
        client.chat.completions.create = delayed_create
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)
    return model_name

_SYSTEM_INSTRUCTIONS = """\
あなたは交通安全領域の高度な専門データサイエンティストです。
交通事故統計データベース（DuckDB）を使い、SQLとPythonを駆使して分析を行います。

## データ基盤
- **テーブル名**: 全てのデータは `accidents` という1つのテーブルに格納されています。
- **データの内容**: 2020年と2024年の交通事故原票データ（約60万件）が含まれています。
- **メタデータ**: `get_semantic_catalog` でカラム名や計算式を確認できます。

## 鉄の掟（データ完全性の遵守）
1. **捏造の厳禁**: SQLがエラーになったり結果が0件の場合、絶対に「仮のデータ」や「業界平均」を捏造してはいけません。
2. **ファイルアクセスの禁止**: `READ_CSV` や `open()` でローカルCSVを直接読もうとしないでください。全てのデータは `accidents` テーブルにあります。
3. **事実に基づく報告**: 取得できたデータのみを使い、不明な点は「不明」と正しく報告してください。

## 分析の標準ワークフロー
1. **カタログ確認**: `get_semantic_catalog` で利用可能なカラムと指標の定義を確認する。
2. **データ抽出**: `run_traffic_query` で `accidents` テーブルから必要な集計結果を得る。
3. **高度な分析・可視化**: 抽出したデータを `execute_python` に渡し、グラフ作成や統計解析を行う。
4. **統合解釈**: 「数値 -> グラフ -> 専門的解釈 -> 限界点」の順で報告する。

## Python実行 (`execute_python`) の極意
- **データ取得**: `con = duckdb.connect(db_path)` を使い、Python内で直接SQLを投げて DataFrame (`con.execute(sql).df()`) を取得するのが最も安定します。
- **可視化**: グラフは `plt.savefig('static/plots/ファイル名.png')` で保存し、そのファイル名を回答に含めてください。
- **日本語**: `japanize_matplotlib` が有効なので、ラベル等に日本語を自由に使ってください。
- **エラー回避**: 複雑なSQLより、シンプルなSQLでデータを取得し、Pandasで加工する方が成功率が高いです。
"""

def build_agents(model: str | None = None) -> tuple[Agent, Agent]:
    model_name = model or os.getenv("AGENT_MODEL", "gemini-2.5-flash")
    resolved = _resolve_model(model_name)

    analyst = Agent(
        name="TrafficSafetyAnalyst",
        instructions=_SYSTEM_INSTRUCTIONS,
        tools=ALL_TOOLS,
        model=resolved,
    )

    return None, analyst
