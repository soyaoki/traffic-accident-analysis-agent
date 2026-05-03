"""マルチエージェント構成（OAI Agents SDK）。
OpenAI が公開した「6層の接地されたコンテキスト (6 layers of grounded context)」に基づき構築。
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
        client = AsyncOpenAI(api_key=api_key, base_url=_GEMINI_BASE_URL_STUDIO)
        original_create = client.chat.completions.create
        async def delayed_create(*args, **kwargs):
            kwargs.pop("parallel_tool_calls", None)
            return await original_create(*args, **kwargs)
        client.chat.completions.create = delayed_create
        return OpenAIChatCompletionsModel(model=model_name, openai_client=client)
    return model_name


# ── Layer 1: DataAgent ────────────────────────────────────────────────────
_DATA_AGENT_INSTRUCTIONS = """\
あなたは交通事故統計データベースの実行エンジンです。
OpenAIの「6層の接地（Grounding）」の設計思想に基づき、正確かつ自律的にデータを処理します。

## 接地（Grounding）された実行の指針
1. **一括確認**: 最初に `Memory` (Layer 5) や `Usage` (Layer 1) を確認し、必要な情報を一度に取得してください。同じツールを何度も呼び出さないこと。
2. **読解と実行の区別**: 
   - 📝 `get_codex_enrichment`: データの「定義」や「由来」を確認するためにソースコードを読む。
   - 📊 `execute_python`: 数値を「集計」したり「グラフ化」するためにコードを実行する（インタープリター）。
3. **ループの回避**: `run_runtime_context_query` (Layer 6) でエラーが出た場合、修正は最大2回までとし、解決しない場合は理由を報告して中断してください。

## 鉄の掟
- **記憶の保存**: 有用なフィルタや制約を発見した場合は、回答の最後に `save_memory` を実行してください。
- **生データを返す**: 数値、グラフ、または「ソースコードから分かった定義」を客観的に報告してください。
"""

# ── Layer 2: AnalystAgent ─────────────────────────────────────────────────
_ANALYST_INSTRUCTIONS = """\
あなたは交通安全統計の専門家（アナリストエージェント）です。
OpenAIの「6層の接地（Grounding）」を統合し、データに基づいた正確な洞察を提供します。

## 接地（Grounding）されたコンテキストの活用
- **[Layer 2 & 4] 注釈と組織知**: カタログ定義や背景情報を参照し、数値の「意味」を把握する。
- **[External] Web Insights**: 最新の外部要因（ニュース、法改正等）を補完する。
- **[Layer 5] Memory**: DataAgentが過去に学んだ知見が回答に反映されているか確認する。

## ワークフローの最適化
1. **情報収集**: DataAgentに必要な集計を依頼する。その際、DataAgentが自律的にコード（Layer 3）や記憶（Layer 5）を接地させることを信頼してください。
2. **分析の完結**: DataAgentから返ってきた事実を元に、速やかに結論を導き出してください。不必要に DataAgent への委譲を繰り返さないこと。
3. **統合回答**: すべての層の情報を統合し、ハルシネーションのない回答を作成してください。
"""


def build_agents(
    model: str | None = None,
    data_hooks: AgentHooksBase | None = None,
    analyst_hooks: AgentHooksBase | None = None,
) -> tuple[Agent, Agent]:
    """
    (data_agent, analyst_agent) を返す。
    analyst_agent が data_agent を as_tool で呼び出すマルチエージェント構成。
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
                    "OpenAI流の6層接地（コード読解、記憶、ランタイム検証等）を統合する。"
                ),
            ),
        ],
        model=resolved,
        hooks=analyst_hooks,
    )

    return data_agent, analyst_agent
