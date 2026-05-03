import streamlit as st
import asyncio
import os
import json
import traceback
from typing import Any
from dotenv import load_dotenv

load_dotenv(override=True)

from agents.tracing import set_tracing_disabled
_has_openai_key = os.getenv("OPENAI_API_KEY", "").startswith("sk-") and len(os.getenv("OPENAI_API_KEY", "")) > 20
if not _has_openai_key:
    set_tracing_disabled(True)

from agents import Runner
from agents.lifecycle import AgentHooksBase
from agents.tool_context import ToolContext
from agents.tool import Tool
from src.agent import build_agents

st.set_page_config(
    page_title="AI Data Agent (OpenAI Philosophy)",
    page_icon="🚗",
    layout="wide",
)

st.sidebar.title("分析設定")
model_option = st.sidebar.selectbox(
    "モデル選択",
    ["gemini-2.5-flash", "gemini-1.5-flash", "gpt-4o-mini", "gpt-4o"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 接地（Grounding）の構成
OpenAIのブログで紹介された「6層のコンテキスト」を、シンプルな構成で再現しています。
1. **使用状況**: 典型的なクエリ例
2. **注釈**: データカタログ定義
3. **コード由来**: 前処理ロジックの読解
4. **組織知**: 専門知識・背景情報の参照
5. **メモリ**: 過去の修正内容の再利用
6. **ランタイム**: ライブクエリと自己修正
""")

st.title("🚗 交通事故統計 AIデータエージェント")
st.caption("OpenAIの「6層の接地」の思想をシンプルに実装した PoC")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for plot_path in message.get("plots", []):
            if os.path.exists(plot_path):
                st.image(plot_path)

if not st.session_state.messages and "pending_prompt" not in st.session_state:
    st.markdown("### 試してみる質問例")
    cols = st.columns(3)
    sample_queries = [
        "2024年に自転車の事故件数や死亡率に変化はある？最新の法改正の影響も踏まえて分析して",
        "2020年から2024年にかけて死亡事故が減っている要因は？社会情勢やニュースと関連づけて考察して",
        "最近の電動キックボード（特定小型原動機付自転車）の事故傾向と、施行されたルールの関係を教えて",
    ]
    for i, query in enumerate(sample_queries):
        if cols[i].button(query):
            st.session_state.pending_prompt = query
            st.rerun()

prompt = st.chat_input("質問を入力してください")
if "pending_prompt" in st.session_state and not prompt:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("エージェントが分析中...", expanded=True)
        result_placeholder = st.empty()
        plots_container = st.container()
        collected_plots: list[str] = []

        class DataAgentHooks(AgentHooksBase):
            async def on_tool_start(self, context: Any, agent: Any, tool: Tool) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "get_table_usage_metadata":
                    status.write("📖 **Layer 1: Usage (使用状況) を接地中...**")
                elif name == "get_codex_enrichment":
                    status.write("📝 **Layer 3: Code-derived (前処理ロジック) を接地中...**")
                elif name == "get_learned_memory":
                    status.write("🧠 **Layer 5: Memory (過去の知見) を接地中...**")
                elif name == "save_memory":
                    status.write("💾 **Layer 5: 今回の知見を Memory に保存中...**")
                elif name == "run_runtime_context_query":
                    sql = ""
                    if isinstance(context, ToolContext):
                        try:
                            sql = json.loads(context.tool_arguments).get("sql", "")
                        except Exception:
                            pass
                    if sql:
                        status.write(f"🔍 **Layer 6: Runtime (ライブクエリ) 実行中:**\n```sql\n{sql}\n```")
                    else:
                        status.write("🔍 **Layer 6: Runtime 実行中...**")
                elif name == "execute_python":
                    status.write("📊 **分析実行中 (Python インタープリター)...**")

            async def on_tool_end(self, context: Any, agent: Any, tool: Tool, result: str) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "run_runtime_context_query":
                    try:
                        rows = json.loads(result)
                        count = len(rows) if isinstance(rows, list) else "?"
                        status.write(f"✅ **クエリ完了** — {count} 件取得")
                    except Exception:
                        pass
                elif name == "execute_python":
                    try:
                        data = json.loads(result)
                        plots = data.get("plots", [])
                        for p in plots:
                            if p not in collected_plots:
                                collected_plots.append(p)
                                with plots_container:
                                    st.image(p, caption=os.path.basename(p))
                                status.write(f"🖼️ **グラフ生成**: `{os.path.basename(p)}`")
                    except Exception:
                        pass

        class AnalystAgentHooks(AgentHooksBase):
            async def on_tool_start(self, context: Any, agent: Any, tool: Tool) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "query_data":
                    status.write("🤖 **DataAgent への接地を依頼中...**")
                elif name == "google_web_search":
                    query = ""
                    if isinstance(context, ToolContext):
                        try:
                            query = json.loads(context.tool_arguments).get("query", "")
                        except Exception:
                            pass
                    if query:
                        status.write(f"🌐 **External Insights (最新情勢の調査) を実行中:**\n> {query}")
                    else:
                        status.write("🌐 **External Insights を実行中...**")
                elif name == "get_institutional_knowledge":
                    status.write("📖 **Layer 4: Institutional (専門知識) を接地中...**")

            async def on_tool_end(self, context: Any, agent: Any, tool: Tool, result: str) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "google_web_search":
                    status.write("✅ **Web Insights 完了**")
                elif name == "query_data":
                    import re
                    plot_paths = re.findall(r"!\[.*?\]\((static/plots/.*?\.png)\)", result)
                    for p in plot_paths:
                        if p not in collected_plots:
                            collected_plots.append(p)
                            with plots_container:
                                st.image(p, caption=os.path.basename(p))
                            status.write(f"🖼️ **グラフ生成**: `{os.path.basename(p)}`")

        try:
            async def run_agent():
                _, analyst = build_agents(
                    model=model_option,
                    data_hooks=DataAgentHooks(),
                    analyst_hooks=AnalystAgentHooks(),
                )
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                # 思考の深さに応じてターン数を調整 (SDKデフォルトに依存)
                return await Runner.run(analyst, history)

            result = asyncio.run(run_agent())
            final_output = result.final_output or ""
            
            result_placeholder.markdown(final_output)
            status.update(label="分析完了 ✅", state="complete", expanded=False)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_output,
                "plots": collected_plots,
            })

        except Exception as e:
            status.update(label="エラー発生", state="error", expanded=True)
            st.error(f"エラー: {e}")
            st.code(traceback.format_exc())
