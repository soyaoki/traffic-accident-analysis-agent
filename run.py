"""デモ実行スクリプト。
"""

import argparse
import asyncio
import os
import json

from dotenv import load_dotenv
load_dotenv()

from agents.tracing import set_tracing_disabled
_has_openai_key = bool(os.getenv("OPENAI_API_KEY", "").startswith("sk-")) and len(os.getenv("OPENAI_API_KEY", "")) > 10
if not _has_openai_key:
    set_tracing_disabled(True)

from agents import Runner
from src.agent import build_agents

async def run_query(query: str) -> str:
    """エージェントを実行して結果を返す。"""
    _, analyst = build_agents()
    
    result = ""
    # stream_events() を使用してイベントを処理
    async for event in Runner.run_streamed(analyst, query).stream_events():
        # Agent からのテキスト差分を表示
        if event.type == "run_item_delta":
            if event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
                result += event.delta.text
        
        # ツール呼び出しの開始を表示
        elif event.type == "run_step":
            if event.step.type == "tool_calls":
                for tc in event.step.tool_calls:
                    # エージェントがツールを呼ぶ際に出力
                    print(f"\n[Tool Call] {tc.tool_name}")
    print("\n")
    return result

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="2020年と2024年の死亡事故件数の変化を教えてください。")
    args = parser.parse_args()

    model = os.getenv("AGENT_MODEL", "gemini-2.5-flash")
    print(f"モデル: {model}\n")

    await run_query(args.query)

if __name__ == "__main__":
    asyncio.run(main_async())
