"""マルチエージェント構成（OAI Agents SDK）。

2層構造:
  DataQueryAgent     — データ層。pandas ツールだけを持つ純粋なデータ実行エージェント。
  TrafficSafetyAnalyst — 分析層。DataQueryAgent を tool として呼び出し、
                         カタログ(Layer 2)の文脈で結果を解釈・統合する。

モデル切り替えは環境変数 AGENT_MODEL で行う（デフォルト: gpt-4o-mini）。
"""

import os
from pathlib import Path

from agents import Agent

from src.tools import ALL_TOOLS

_CATALOG_PATH = Path(__file__).parent / "context" / "catalog.yaml"


def _load_catalog() -> str:
    return _CATALOG_PATH.read_text(encoding="utf-8")


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
    _model = model or os.getenv("AGENT_MODEL", "gpt-4o-mini")

    # Layer 1: データ実行エージェント
    data_agent = Agent(
        name="DataQueryAgent",
        instructions=_DATA_AGENT_INSTRUCTIONS,
        tools=ALL_TOOLS,
        model=_model,
    )

    # Layer 2: 分析・統合エージェント（data_agent を tool として利用）
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
        model=_model,
    )

    return data_agent, analyst_agent
