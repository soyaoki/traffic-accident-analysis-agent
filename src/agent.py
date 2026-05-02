"""マルチエージェント構成（OAI Agents SDK）。

2層構造:
  DataQueryAgent       — データ層。pandas ツールだけを持つ純粋なデータ実行エージェント。
  TrafficSafetyAnalyst — 分析層。DataQueryAgent を tool として呼び出し、
                         カタログ(Layer 2)の文脈で結果を解釈・統合する。

対応モデル（AGENT_MODEL 環境変数で切り替え）:
  OpenAI:  gpt-4o-mini / gpt-4o           → OPENAI_API_KEY
  Gemini (AI Studio): gemini-2.5-flash / gemini-1.5-flash → GEMINI_API_KEY
  Gemini (Vertex AI): google/gemini-1.5-flash             → GEMINI_API_KEY
"""

import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from src.tools import ALL_TOOLS

# システム環境変数を優先
load_dotenv(override=True)

_CATALOG_PATH = Path(__file__).parent / "context" / "catalog.yaml"

_GEMINI_BASE_URL_STUDIO = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_PREFIXES = ("gemini-", "google/gemini-")


def _load_catalog() -> str:
    return _CATALOG_PATH.read_text(encoding="utf-8")


def _resolve_model(model_name: str) -> str | OpenAIChatCompletionsModel:
    """モデル名からエージェントに渡す model オブジェクトを返す。"""
    if any(model_name.startswith(p) for p in _GEMINI_PREFIXES):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY が未設定です。")

        # Vertex AI (google/ プレフィックス) の場合はベースURLを動的に構成
        if model_name.startswith("google/"):
            project_id = os.getenv("VERTEX_PROJECT_ID")
            region = os.getenv("VERTEX_REGION", "us-central1")
            if not project_id:
                raise ValueError("Vertex AI 使用時は VERTEX_PROJECT_ID が必要です。")
            
            # Vertex AI OpenAI-compatible endpoint
            base_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/endpoints/openapi"
            wait_time = 0.2 # 有料枠なので最小限
        else:
            base_url = _GEMINI_BASE_URL_STUDIO
            wait_time = 2.0 # 無料枠 (AI Studio) 用のレート制限

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # レート制限対策のラッパー
        original_create = client.chat.completions.create
        async def delayed_create(*args, **kwargs):
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            return await original_create(*args, **kwargs)
        client.chat.completions.create = delayed_create

        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)

    # OpenAI（デフォルト）
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY が未設定です。")
    return model_name


# ── Layer 1: DataQueryAgent ────────────────────────────────────────────────
_DATA_AGENT_INSTRUCTIONS = """\
あなたは交通事故統計データの実行エンジンです。
渡された指示に従い、提供されたツールを使ってデータを取得・集計してください。

ルール:
- 複数のツールが必要な場合はすべて呼び出してから結果をまとめること
- 数値は加工せずそのまま返すこと（解釈は上位エージェントが行う）
- ツールが返す JSON をそのまま含めること
"""

# ── Layer 2: TrafficSafetyAnalyst ─────────────────────────────────────────
_ANALYST_INSTRUCTIONS_WITH_CONTEXT = """\
あなたは交通安全領域のデータサイエンティストです。
`query_traffic_data` ツールを通じてデータを取得し、以下のカタログと分析方針に基づいて解釈・統合します。

## データカタログ（Layer 2: 人間による注釈）

{catalog}

## 分析の指針
- 数値は「絶対値」と「変化率」をセットで述べること
- サポカー分析では必ず「残存事故バイアス」の留保を付けること
- 2020年データにはサポカー列が存在しない旨を明示すること
- 横断面データの限界（防いだ事故は観測不能）を明示すること
- 結果は「数値 → 解釈 → 限界点」の3点構造で提示すること
"""

_ANALYST_INSTRUCTIONS_NO_CONTEXT = """\
あなたは交通事故統計データの分析エージェントです。
`query_traffic_data` ツールを通じてデータを取得し、質問に答えてください。
結果は日本語で、数値と解釈をセットで提示してください。
"""


def build_agents(
    with_context: bool = True,
    model: str | None = None,
) -> tuple[Agent, Agent]:
    """
    (data_agent, analyst_agent) のタプルを返す。
    analyst_agent が data_agent を tool として呼び出すマルチエージェント構成。
    """
    model_name = model or os.getenv("AGENT_MODEL", "gpt-4o-mini")
    resolved = _resolve_model(model_name)

    data_agent = Agent(
        name="DataQueryAgent",
        instructions=_DATA_AGENT_INSTRUCTIONS,
        tools=ALL_TOOLS,
        model=resolved,
    )

    if with_context:
        catalog = _load_catalog()
        analyst_instructions = _ANALYST_INSTRUCTIONS_WITH_CONTEXT.format(catalog=catalog)
        analyst_name = "TrafficSafetyAnalyst_WithContext"
    else:
        analyst_instructions = _ANALYST_INSTRUCTIONS_NO_CONTEXT
        analyst_name = "TrafficSafetyAnalyst_NoContext"

    analyst_agent = Agent(
        name=analyst_name,
        instructions=analyst_instructions,
        tools=[
            data_agent.as_tool(
                tool_name="query_traffic_data",
                tool_description=(
                    "交通事故統計データの集計・分析を実行する。"
                    "2020年・2024年の比較、死亡事故セグメント分析、"
                    "サポカー効果分析、2030年予測などが可能。"
                ),
            )
        ],
        model=resolved,
    )

    return data_agent, analyst_agent
