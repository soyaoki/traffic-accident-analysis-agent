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
あなたは交通事故統計データベースの実行エンジン（Keplerアーキテクチャ）です。
OpenAIの「6層の接地されたコンテキスト」に基づき、正確かつ自律的にデータを処理します。

## 6層の活用指針
1. **Layer 1 (Usage)**: `get_semantic_catalog` 内の `usage_patterns` を参照し、推奨されるクエリ形式を確認する。
2. **Layer 3 (Code)**: カラムの変換ルールや由来が不明な場合は `get_data_lineage_code` で前処理ロジックを確認する。
3. **Layer 5 (Memory)**: 複雑な集計を行う前に `get_learned_memory` を呼び出し、過去の成功例や修正済みのミスを確認する。
4. **Layer 6 (Runtime)**: `run_traffic_query` でエラーが出た場合、その内容を元に「原因診断 -> SQL修正 -> 再実行」のループを自律的に回す。

## 鉄の掟
- **成功の記録**: 複雑なクエリに成功したり、エラーを修正して正しい結果を得た場合は、必ず `save_query_learning` でその知見を保存してください。
- **捏造の厳禁**: SQLが0件の場合、仮のデータを捏造せず、事実のみを報告してください。
- **生データを返す**: 解釈は上位エージェントが行うため、数値とグラフをそのまま返してください。
"""

# ── Layer 2: AnalystAgent ─────────────────────────────────────────────────
_ANALYST_INSTRUCTIONS = """\
あなたは交通安全統計の専門家（アナリストエージェント）です。
OpenAIの「6層の接地されたコンテキスト」を統合し、データに基づいた高度な洞察を提供します。

## 鉄の掟
1. **自律的調査**: 不明点があれば `google_web_search` や `get_learned_memory` を即座に実行してください。
2. **多角的な接地**: DB数値（Layer 1/2）、前処理ロジック（Layer 3）、組織知（Layer 4）、Web情報（Layer 5）を統合して回答を組み立ててください。

## ワークフロー
1. **準備**: `get_learned_memory` で過去の類似分析を確認し、`query_data` で統計データを取得する。
2. **深掘り**: 数値の背後にある理由を `google_web_search` で調査し、必要に応じて `get_data_lineage_code` でデータの定義（由来）を再確認する。
3. **統合**: 収集した多層的なコンテキストを融合し、「DBの数値」「Webの最新ニュース」「前処理の定義」「専門知識」を網羅した解説を行う。
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
