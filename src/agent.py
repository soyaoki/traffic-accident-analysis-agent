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
データマイニングの標準プロセスである **CRISP-DM** に準拠し、エンジニアが取得した生データをドメイン知識（Layer 2 & 4）に基づいて解釈・可視化（Python）します。

## 接地（Grounding）と CRISP-DM の適用
1. **ビジネス/データの理解 (Understanding)**: 分析を開始する前に、必ず `get_human_annotations` (Layer 2) および `get_institutional_knowledge` (Layer 4) の **両方** を実行してください。
2. **データの準備 (Preparation)**: エンジニアへの依頼を通じて必要なデータを揃えます。
3. **評価と解釈 (Evaluation)**: 接地された全レイヤーの情報を統合し、数値の背景を考慮して解釈してください。
4. **可視化と展開**: `execute_python` を用いて、グラフを生成してください。
5. **重要：画像パスの確実な取得と報告**: 
   - `execute_python` の戻り値（JSON）に含まれる `plots` リストから、生成された実際の画像パス（例：`static/plots/...`）を確実に読み取ってください。
   - 取得した実在する画像パスを、**必ず** Markdown形式（`![タイトル](取得した実際のパス)`）で報告に含めてください。**絶対に `markdown_chart_path` などのプレースホルダー（仮の文字列）を使わないでください。** 実際のパスが漏れるとグラフが表示されません。
6. **専門的留保**: 「残存事故バイアス」等の統計的な罠に注意し、深いインサイトを提供してください。
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
チーム（DataAnalyst, DataEngineer）を指揮し、プロフェッショナルな視覚的レポートをユーザーに届けることがあなたの最終責務です。

## 遂行上の絶対ルール（MUST）
1. **必ずWEB検索を実行**: すべてのユーザーリクエストに対し、まず最初に `google_web_search` を実行して最新の法改正や社会情勢を調査してください。データベース内の数字だけでは不完全です。
2. **視覚的プレゼンテーション**: DataAnalystから送られてきた Markdown 形式のグラフ（`![タイトル](パス)`）を、**一切省略せずに**そのまま最終レポートの本文中に適切に配置してください。
3. **外部ソースの常時掲載**: `External Web Insights` で得られたソース情報（URLリンク）は、必ずレポートの末尾に「参照元（External Sources）」として掲載してください。
4. **一気通貫の完結**: 1回の回答で、データ、グラフ、専門的解釈、および参照元をすべて統合したプロフェッショナルなレポートを提示してください。
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
