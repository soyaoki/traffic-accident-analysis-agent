"""マルチエージェント構成（OAI Agents SDK）。

2層構造:
  DataAgent      — データ実行層。SQL・Python・カタログ参照ツールを持つ純粋な実行エージェント。
  AnalystAgent   — 解釈・統合層。DataAgent を as_tool で呼び出し、
                   Layer 2 カタログの文脈で結果を解釈する。

モデル切り替えは環境変数 AGENT_MODEL で行う（デフォルト: gemini-2.5-flash）。
"""

import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent
from agents.lifecycle import AgentHooksBase
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from src.tools import DATA_TOOLS, ANALYST_TOOLS
from src.preprocess import init_db, DB_PATH

load_dotenv(override=True)

_GEMINI_BASE_URL_STUDIO = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_PREFIXES = ("gemini-", "google/gemini-")


def _resolve_model(model_name: str) -> str | OpenAIChatCompletionsModel:
    if any(model_name.startswith(p) for p in _GEMINI_PREFIXES):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY が未設定です。")
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
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            kwargs.pop("parallel_tool_calls", None)
            return await original_create(*args, **kwargs)

        client.chat.completions.create = delayed_create
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)
    return model_name


# ── Layer 1: DataAgent ────────────────────────────────────────────────────
_DATA_AGENT_INSTRUCTIONS = """\
あなたは交通事故統計データベースの実行エンジンです。
上位エージェントの指示に従い、ツールを使ってデータを取得・集計・可視化してください。

## 鉄の掟
1. **捏造の厳禁**: SQLがエラーになったり結果が0件の場合、仮のデータや業界平均を捏造しないこと
2. **ファイルアクセスの禁止**: `READ_CSV` や `open()` は使わず、必ず `accidents` テーブルを参照すること
3. **生データを返す**: 解釈・考察は行わず、取得した数値・グラフをそのまま返すこと（解釈は上位エージェントが行う）

## 標準ワークフロー
1. `get_semantic_catalog` でカラム定義を確認する
2. `run_traffic_query` で集計データを取得する
3. 必要に応じて `execute_python` でグラフ・統計解析を実行する
"""

# ── Layer 2: AnalystAgent ─────────────────────────────────────────────────
_ANALYST_INSTRUCTIONS = """\
あなたは交通安全統計の専門家（アナリストエージェント）です。
`query_data` でデータを取得し、ドメイン知識と背景知識を統合して、数値の背後にある意味を解説します。

## 知識の使い分け
- `get_domain_knowledge`: 統計的バイアス、分析方針、交通安全学としての専門知識
- `get_background_knowledge`: 社会情勢（コロナ等）、法改正、データの収集背景（Off-Data コンテキスト）

## ワークフロー
1. 知識の確認: `get_domain_knowledge` と `get_background_knowledge` を参照し、分析の前提条件を把握する
2. データ取得: `query_data` を使って、必要な集計や可視化を実行する
3. 解釈と統合: 数値の変化が「施策の効果」なのか「外的要因（背景知識）」や「統計上の偏り（ドメイン知識）」なのかを多角的に分析し、回答を組み立てる
"""


def build_agents(
    model: str | None = None,
    data_hooks: AgentHooksBase | None = None,
    analyst_hooks: AgentHooksBase | None = None,
) -> tuple[Agent, Agent]:
    """
    (data_agent, analyst_agent) を返す。
    analyst_agent が data_agent を as_tool で呼び出すマルチエージェント構成。
    初回呼び出し時に DuckDB を初期化する。
    """
    if not DB_PATH.exists():
        init_db()

    model_name = model or os.getenv("AGENT_MODEL", "gemini-2.5-flash")
    resolved = _resolve_model(model_name)

    data_agent = Agent(
        name="DataAgent",
        instructions=_DATA_AGENT_INSTRUCTIONS,
        tools=DATA_TOOLS,
        model=resolved,
        hooks=data_hooks,
    )

    analyst_agent = Agent(
        name="AnalystAgent",
        instructions=_ANALYST_INSTRUCTIONS,
        tools=[
            *ANALYST_TOOLS,
            data_agent.as_tool(
                tool_name="query_data",
                tool_description=(
                    "交通事故統計データの取得・集計・可視化を実行する。"
                    "SQL クエリ、Python 統計解析、グラフ生成が可能。"
                ),
            ),
        ],
        model=resolved,
        hooks=analyst_hooks,
    )

    return data_agent, analyst_agent
