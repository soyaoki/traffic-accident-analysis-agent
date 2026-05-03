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

from src.tools import (
    ENGINEER_TOOLS,
    ANALYST_TOOLS,
    MANAGER_TOOLS,
    SCIENTIST_TOOLS,
)
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


# ── Layer 1: DataEngineer ────────────────────────────────────────────────
_ENGINEER_INSTRUCTIONS = """\
あなたは交通事故統計データの「データエンジニア」です。
データの所在（Layer 1）、由来（Layer 3）、過去の知見（Layer 5）、および実行環境（Layer 6）に責任を持ちます。

## 接地（Grounding）の義務
1. **事前の全層確認**: SQLを構築・実行する前に、必ず `get_table_usage_metadata` (Layer 1), `get_codex_enrichment` (Layer 3), `get_learned_memory` (Layer 5) の **3つすべて** を実行し、情報を統合してください。どれか一つでも欠けると接地が不完全とみなされます。
2. **自己修復**: SQLエラーが発生した場合は `Runtime` (Layer 6) の情報を元に自律的に修正してください。
3. **知見の保存**: 修正に成功したり、有益な制約を発見した場合は、必ず `save_memory` (Layer 5) を実行してください。

生データ（数値）と客観的な事実のみを上位エージェントに報告してください。
"""

# ── Layer 2: DataAnalyst ─────────────────────────────────────────────────
_ANALYST_INSTRUCTIONS = """\
あなたは交通事故統計の「データアナリスト」です。
エンジニアが取得した生データを、ドメイン知識（Layer 2 & 4）に基づいて解釈し、可視化（Python）を行います。

## 接地（Grounding）の義務
1. **事前の全層確認**: 解釈を開始する前に、必ず `get_human_annotations` (Layer 2) と `get_institutional_knowledge` (Layer 4) の **両方** を実行してください。指標の定義と専門知識の両方を接地させることが必須です。
2. **統計的解釈**: 接地された知識に基づき、数値の背景（法改正、コロナ影響等）を考慮して解釈してください。
3. **可視化**: 分析結果を直感的に理解できるよう、`execute_python` を用いて適切なグラフを生成してください。
4. **専門的留保**: 「残存事故バイアス」等の統計的な罠に注意し、単なる数値の比較に留まらない洞察を提供してください。
"""


# ── Layer 3: DataScientist (Future Placeholder) ──────────────────────────
_SCIENTIST_INSTRUCTIONS = """\
あなたは交通事故統計の「データサイエンティスト」です。
現在は準備段階ですが、将来的に予測モデリングや高度な統計検定を担当します。
今回のタスクでは、必要に応じてアナリストをサポートしてください。
"""

# ── Layer 4: Manager ─────────────────────────────────────────────────────
_MANAGER_INSTRUCTIONS = """\
あなたは交通安全分析プロジェクトの「マネージャー」です。
チーム（DataAnalyst, DataEngineer）を指揮し、最終的な統合報告書をユーザーに届けることがあなたの最終責務です。

## 回答のスタイル：Senior Executive Summary
1. **簡潔さと高信号**: 回答は冗長さを排し、事実とインサイトを凝縮した「エグゼクティブ・サマリー」形式で提供してください。不必要な前置きや長文の解説は不要です。
2. **構造化**: 結論、データ（数値・グラフ）、解釈、限界事項を箇条書きや簡潔な見出しで整理してください。
3. **中途報告の禁止**: 分析結果を含まない回答をユーザーに返してはなりません。必ずチームからの結果を統合して回答してください。
4. **一気通貫の回答**: 1回の回答で、すべての情報を統合した完成されたレポートを作成してください。
"""


def build_agents(
    model: str | None = None,
    engineer_hooks: AgentHooksBase | None = None,
    analyst_hooks: AgentHooksBase | None = None,
    manager_hooks: AgentHooksBase | None = None,
) -> tuple[Agent, Agent, Agent, Agent]:
    """
    (engineer, analyst, scientist, manager) を返す。
    Manager -> Analyst -> Engineer の順で呼び出す階層構造。
    """
    if not DB_PATH.exists():
        init_db()

    model_name = model or os.getenv("AGENT_MODEL", "gemini-2.5-flash")
    resolved = _resolve_model(model_name)

    # 1. DataEngineer
    engineer = Agent(
        name="DataEngineer",
        instructions=_ENGINEER_INSTRUCTIONS,
        tools=ENGINEER_TOOLS,
        model=resolved,
        hooks=engineer_hooks,
    )

    # 2. DataAnalyst
    analyst = Agent(
        name="DataAnalyst",
        instructions=_ANALYST_INSTRUCTIONS,
        tools=[
            *ANALYST_TOOLS,
            engineer.as_tool(
                tool_name="request_data_retrieval",
                tool_description="データエンジニアにSQLによる生データの取得、スキーマ確認、メモリ操作を依頼する。",
            ),
        ],
        model=resolved,
        hooks=analyst_hooks,
    )

    # 3. DataScientist (Placeholder)
    scientist = Agent(
        name="DataScientist",
        instructions=_SCIENTIST_INSTRUCTIONS,
        tools=SCIENTIST_TOOLS,
        model=resolved,
    )

    # 4. Manager
    manager = Agent(
        name="Manager",
        instructions=_MANAGER_INSTRUCTIONS,
        tools=[
            *MANAGER_TOOLS,
            analyst.as_tool(
                tool_name="request_analysis",
                tool_description="データアナリストに専門的な分析、解釈、および可視化（グラフ生成）を依頼する。",
            ),
        ],
        model=resolved,
        hooks=manager_hooks,
    )

    return engineer, analyst, scientist, manager
