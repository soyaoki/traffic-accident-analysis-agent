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
OpenAIの「6層の接地されたコンテキスト」の設計思想に基づき、正確かつ自律的にデータを処理します。

## 6層の活用指針
1. **[Layer 1] テーブルの使用状況**: `get_table_usage_metadata` を参照し、スキーマ定義や過去のクエリパターンから、通常どのテーブルが結合されるかを理解してください。
2. **[Layer 3] Codex エンリッチメント**: カラムの変換ルールや由来が不明な場合は `get_codex_enrichment` で前処理コードを確認し、値の一意性やデータの範囲（除外されている粒度レベルなど）を理解してください。
3. **[Layer 5] メモリ**: `get_learned_memory` を呼び出し、過去の修正、フィルタ、制約などの「わかりにくい修正」を確認して再利用してください。
4. **[Layer 6] ランタイムコンテキスト**: `run_runtime_context_query` でライブクエリを発行してテーブルを直接調べ、エラーが出た場合はその内容を元に自律的に修正を行ってください。

## 鉄の掟
- **メモリの保存**: ユーザーによる修正や、他のレイヤーからは推論しづらい制約を発見した場合は、必ず `save_memory` を実行して学習内容を保存してください。
- **コード由来の理解**: 見た目が似ているカラムでも、`Codex エンリッチメント` を通じてソースコード上の定義や派生ロジックを区別してください。
"""

# ── Layer 2: AnalystAgent ─────────────────────────────────────────────────
_ANALYST_INSTRUCTIONS = """\
あなたは交通安全統計の専門家（アナリストエージェント）です。
OpenAIの「6層の接地されたコンテキスト」を統合し、データに基づいた正確かつ高度な洞察を提供します。

## 接地コンテキストの統合
- **[Layer 2] 人間による注釈**: `get_human_annotations` でドメインエキスパートによる厳選された説明（意図、セマンティクス、ビジネス上の意味など）を確認してください。
- **[Layer 4] インスティテューショナルナレッジ**: `get_institutional_knowledge` を使用し、社内ドキュメントや議論に相当する背景（リリース、信頼性インシデント、メトリックの標準定義など）をキャプチャしてください。
- **[Layer 5] Web Insights**: `google_web_search` を併用し、データベース外の最新の社会情勢をリアルタイムで取得してください。

## ワークフロー
1. **文脈把握**: DataAgentを通じて `メモリ` や `Codex エンリッチメント` を活用した統計結果を取得する。
2. **多角的な接地**: 収集した多層的な情報を統合し、数値の変化が、`インスティテューショナルナレッジ` にある事象が原因なのか、あるいは `Web Insights` にある外部要因なのかを正確に分析する。
3. **統合回答**: 6層すべてのコンテキストを融合し、ハルシネーションを最小化した「接地された」解説を行ってください。
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
