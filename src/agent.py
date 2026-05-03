"""マルチエージェント構成（OAI Agents SDK）。
OpenAI が公開した「6層の接地されたコンテキスト (6 layers of grounded context)」に基づき構築。
"""

import os
import asyncio
from pydantic import BaseModel

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


class PlotInfo(BaseModel):
    path: str
    title: str

class SourceInfo(BaseModel):
    url: str
    title: str

class FinalReport(BaseModel):
    report: str
    plots: list[PlotInfo]
    sources: list[SourceInfo]


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
データマイニングの標準プロセス **CRISP-DM** に準拠し、生データを解釈・可視化し、構造化された結果をマネージャーに報告します。

## ワークフロー（スキップ厳禁）
1. **理解**: `get_human_annotations` (L2) と `get_institutional_knowledge` (L4) を実行し、ドメイン知識を接地します。
2. **準備**: **必ず** `request_data_retrieval` を使い、エンジニアから必要なデータを取得してください。知識だけで推測することは禁止です。
3. **評価**: 取得した生データとドメイン知識を統合し、インサイトを導出します。
4. **可視化**: **必ず** `execute_python` を使いグラフを生成してください。Pythonコード内では必ず `plt.savefig("static/plots/任意のファイル名.png")` を実行し、画像を保存してください。グラフなしの報告は認められません。
5. **報告**: **必ず**以下のJSONフォーマットでマネージャーに報告してください。Markdownや他の形式は不可です。

```json
{
  "analysis_summary": "ここに、データに基づく分析結果や考察をMarkdown形式（見出し、箇条書きを活用）で記述します。",
  "plots": [
    {
      "path": "static/plots/xxx.png",
      "title": "グラフのタイトル"
    }
  ]
}
```
- `execute_python` から返されるJSON内の `plots` 配列の情報を、上記 `plots` に含めてください。
"""


# ── Layer 3: DataScientist (Future Placeholder) ──────────────────────────
_SCIENTIST_INSTRUCTIONS = """\
あなたは交通事故統計の「データサイエンティスト」です。
現在は準備段階ですが、将来的に予測モデリングや高度な統計検定を担当します。
今回のタスクでは、必要に応じてアナリストをサポートしてください。
"""

# ── Layer 4: Manager ─────────────────────────────────────────────────────
_MANAGER_INSTRUCTIONS = """\
あなたは交通安全分析プロジェクトの「マネージャー」です。チームを指揮し、最終的な分析結果を **厳格なJSON形式** で出力することがあなたの唯一の責務です。

## ワークフローと最終出力形式
1. **情報収集**:
   - まず `google_web_search` を実行し、外部情報を取得します。ツールからはJSON（`summary`と`sources`）が返されます。
   - 次に `request_analysis` を実行し、アナリストから分析結果を取得します。ツールからはJSON（`analysis_summary`と`plots`）が返されます。
2. **情報集約とレポート作成**:
   - `google_web_search` の `summary` と、アナリストの `analysis_summary` を統合し、ユーザーにとって読みやすく専門的な **Markdown形式のレポート本文** を作成してください。見出しや箇条書きを適切に使い、洗練された構成にしてください。
3. **最終出力 (Final Output)**:
   - **他のフォーマットは絶対に許可されません。** 以下のJSONスキーマに厳密に従って、最終的な回答を生成してください。

```json
{
  "report": "ここに、あなたが作成した最終的なレポート本文をMarkdown形式で記述します。",
  "plots": [
    {
      "path": "static/plots/xxx.png",
      "title": "アナリストが報告したグラフのタイトル"
    }
  ],
  "sources": [
    {
      "url": "https://...",
      "title": "Web検索で得られた参照元サイトのタイトル"
    }
  ]
}
```
- アナリストから受け取った `plots` 配列をそのまま最終出力の `plots` に含めてください。
- Web検索で得られた `sources` 配列をそのまま最終出力の `sources` に含めてください。
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
