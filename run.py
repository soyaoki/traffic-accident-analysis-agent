"""デモ実行スクリプト。

使い方:
  uv run python run.py                    # コンテキストありで3問実行
  uv run python run.py --no-context       # コンテキストなしで比較
  uv run python run.py --query "質問文"   # 任意の質問
  uv run python run.py --compare          # コンテキストあり/なしを同じ質問で比較
"""

import argparse
import asyncio
import os
import time

from dotenv import load_dotenv
load_dotenv()

# Gemini使用時やOpenAIキーがプレースホルダの場合は、agentsのトレース送信を最優先で無効化
from agents.tracing import set_tracing_disabled
_has_openai_key = bool(os.getenv("OPENAI_API_KEY", "").startswith("sk-")) and len(os.getenv("OPENAI_API_KEY", "")) > 10
if not _has_openai_key:
    set_tracing_disabled(True)

from agents import Runner
from src.agent import build_agents

# Gemini 無料枠は 5 req/分。クエリ間の待機秒数（余裕を持って 30秒）
_QUERY_INTERVAL_SEC = int(os.getenv("QUERY_INTERVAL_SEC", "30"))

DEMO_QUERIES = [
    "2020年と2024年の死亡事故件数・死者数を比較し、「2030年半減」目標の現在地を整理してください。",
    "死亡負担が最も大きく、かつ削減が停滞しているシナリオはどれですか？ADASが効きにくい構造的な理由も説明してください。",
    "このままのペースで2030年目標を達成できますか？試算してください。",
]

COMPARE_QUERY = "死亡負担が最も大きいシナリオを特定し、サポカーの効果を分析してください。"


def _hr(title: str) -> None:
    print(f"\n{'='*64}")
    print(f"  {title}")
    print("=" * 64)


async def run_query(query: str, with_context: bool) -> str:
    _, analyst = build_agents(with_context=with_context)
    result = await Runner.run(analyst, query)
    return result.final_output


async def run_demo(with_context: bool) -> None:
    label = "コンテキストあり" if with_context else "コンテキストなし"
    for i, query in enumerate(DEMO_QUERIES, 1):
        _hr(f"Q{i} [{label}]")
        print(f"質問: {query}\n")
        result = await run_query(query, with_context=with_context)
        print(result)
        if i < len(DEMO_QUERIES) and _QUERY_INTERVAL_SEC > 0:
            print(f"\n⏳ レートリミット回避のため {_QUERY_INTERVAL_SEC}秒 待機中...")
            time.sleep(_QUERY_INTERVAL_SEC)


async def run_compare() -> None:
    _hr(f"比較クエリ: {COMPARE_QUERY}")

    print("\n▼ コンテキストなし（スキーマのみ）")
    out_no = await run_query(COMPARE_QUERY, with_context=False)
    print(out_no)

    if _QUERY_INTERVAL_SEC > 0:
        print(f"\n⏳ {_QUERY_INTERVAL_SEC}秒 待機中...")
        time.sleep(_QUERY_INTERVAL_SEC)

    print("\n▼ コンテキストあり（Layer 2 カタログ注入）")
    out_yes = await run_query(COMPARE_QUERY, with_context=True)
    print(out_yes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    model = os.getenv("AGENT_MODEL", "gemini-2.5-flash")
    print(f"モデル: {model}")

    if args.compare:
        asyncio.run(run_compare())
    elif args.query:
        result = asyncio.run(run_query(args.query, with_context=not args.no_context))
        _hr("回答")
        print(result)
    else:
        asyncio.run(run_demo(with_context=not args.no_context))


if __name__ == "__main__":
    main()
