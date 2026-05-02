"""マルチエージェント構成（OAI Agents SDK）。
OpenAI Kepler アーキテクチャに基づき、6層の接地コンテキストを統合します。
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

## 6層の活用指針（Kepler Workflow）
1. **[Layer 1] Table Usage**: `get_table_usage_metadata` を参照し、過去の成功クエリや通常どのテーブルが結合されるかの推論を確認する。
2. **[Layer 3] Codex Enrichment**: カラムの変換ルール、値の一意性、除外されている粒度レベル（例：テストデータ除外等）を `get_codex_enrichment` で前処理コードから直接読み取って理解する。
3. **[Layer 5] Memory**: `get_learned_memory` を呼び出し、過去に発見された「わかりにくい修正、フィルタ、制約」を取得して再利用する。
4. **[Layer 6] Runtime Context**: `run_runtime_context_query` を発行してスキーマを検証し、エラーが出た場合はそのフィードバックを元に自力で修正する。

## 鉄の掟
- **記憶の保存**: 新しい修正やフィルタリングの制約を発見した場合は、必ず `save_memory` を実行して将来のベースラインを強化してください。
- **コード由来の理解**: 見た目が似ているカラムでも、`Codex Enrichment` を通じてその由来（ソースコード上の定義）を区別してください。
"""

# ── Layer 2: AnalystAgent ─────────────────────────────────────────────────
_ANALYST_INSTRUCTIONS = """\
あなたは交通安全統計の専門家（アナリストエージェント）です。
OpenAIの「6層の接地されたコンテキスト」を統合し、データに基づいた高度な洞察を提供します。

## 接地コンテキストの統合
- **[Layer 2] Human Annotations**: `get_human_annotations` でスキーマからは推測できない意図やビジネス上のセマンティクスを確認する。
- **[Layer 4] Institutional Knowledge**: `get_institutional_knowledge` を使用し、Slackやドキュメントに相当する背景（リリース、信頼性インシデント等）をキャプチャする。
- **[Layer 5] Web Insights**: `google_web_search` を併用し、データベース外の最新の社会動向（法改正ニュース等）をリアルタイムで取得する。

## ワークフロー
1. **文脈把握**: DataAgentを通じて `Memory` や `Codex Enrichment` を活用した統計結果を取得する。
2. **理由の解明**: 数値の変化（例：特定時期の減少）が、`Institutional Knowledge` にある障害が原因なのか、あるいは `Web Insights` にある法改正の影響なのかを多角的に分析する。
3. **統合回答**: 6層すべてのコンテキストを融合し、単なる集計を超えた「意味のある解説」を行う。
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
                    "Kepler 6-Layer Architecture に基づき、コード、記憶、実行時コンテキストを統合する。"
                ),
            ),
        ],
        model=resolved,
        hooks=analyst_hooks,
    )

    return data_agent, analyst_agent
