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
集計データ（事実）と、最新の社会情勢や専門知識（解釈）を統合して回答します。

## 分析の優先順位
1. **DBデータ（最優先）**: まず `query_data` を使用して、交通事故統計の数値・傾向を把握してください。これがあなたの分析の「主軸」です。
2. **Web検索（補完）**: DBから得られた数値の変化（例：2024年に急減している、等）について、その「理由」を推測・裏付けするために `google_web_search` を使用してください。

## 知識の使い分け
- `query_data`: 実際の事故件数、負傷者数、発生状況などの「生データ」を取得する。
- `google_web_search`: 最新の法改正、社会的なニュース、データの背景にある「外的要因」を調査する。
- `get_domain_knowledge`: 統計的バイアス、分析方針など、交通安全学としての専門知識。
- `get_background_knowledge`: 過去の社会情勢（コロナ等）やデータの収集背景。

## ワークフロー
1. **現状把握**: `query_data` で要求された統計データを集計・可視化する。
2. **疑問の抽出**: 数値に顕著な変化や異常値がある場合、「なぜこの時期に増減したのか？」という疑問を特定する。
3. **外部調査**: `google_web_search` を使い、その時期に施行された法改正や社会的な出来事を調査する。
4. **統合と考察**: DBの数値事実とWebの背景情報を統合し、「2024年に事故が減ったのは、○月の法改正による取り締まり強化が影響している可能性がある」といった説得力のある解説を行う。
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
