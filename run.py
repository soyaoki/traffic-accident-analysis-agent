"""デモ実行スクリプト。

使い方:
  uv run python run.py                           # デフォルト質問
  uv run python run.py --query "質問文"          # 任意の質問
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

from agents.tracing import set_tracing_disabled

# OpenAI キーが実在しない場合はトレース無効
_openai_key = os.getenv("OPENAI_API_KEY", "")
if not (_openai_key.startswith("sk-") and len(_openai_key) > 20):
    set_tracing_disabled(True)

from agents import Runner
from src.agent import build_agents


async def run_query(query: str) -> str:
    _data, analyst = build_agents()
    result = await Runner.run(analyst, query)
    output = result.final_output or ""
    print(output)
    return output


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="2020年と2024年の死亡事故件数の変化を教えてください。",
    )
    args = parser.parse_args()

    model = os.getenv("AGENT_MODEL", "gemini-2.5-flash")
    print(f"モデル: {model}\n")
    print(f"質問: {args.query}\n")
    print("=" * 64)

    await run_query(args.query)


if __name__ == "__main__":
    asyncio.run(main_async())
