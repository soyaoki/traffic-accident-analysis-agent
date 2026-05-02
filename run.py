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
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent

# OpenAI キーが実在しない場合はトレース無効
_openai_key = os.getenv("OPENAI_API_KEY", "")
if not (_openai_key.startswith("sk-") and len(_openai_key) > 20):
    set_tracing_disabled(True)

from openai.types.responses import ResponseTextDeltaEvent
from agents import Runner
from agents.items import ItemHelpers, MessageOutputItem
from src.agent import build_agents


async def run_query(query: str) -> str:
    _, analyst = build_agents()
    output_parts: list[str] = []

    streamed = Runner.run_streamed(analyst, query)
    async for event in streamed.stream_events():

        if isinstance(event, RawResponsesStreamEvent):
            # SDK は responses API を使用。テキストデルタは ResponseTextDeltaEvent.delta に入る
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
                output_parts.append(event.data.delta)

        elif isinstance(event, RunItemStreamEvent):
            if event.name == "tool_called":
                raw = event.item.raw_item
                tool_name = getattr(raw, "name", None) or (raw.get("name", "?") if isinstance(raw, dict) else "?")
                print(f"\n[🔧 {tool_name}]", flush=True)

            elif event.name == "message_output_created" and not output_parts:
                # デルタが来なかった場合の fallback
                if isinstance(event.item, MessageOutputItem):
                    text = ItemHelpers.text_message_output(event.item)
                    print(text, end="", flush=True)
                    output_parts.append(text)

    print()
    return "".join(output_parts)


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
