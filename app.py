import streamlit as st
import asyncio
import os
import json
import traceback
from typing import Any
from dotenv import load_dotenv

load_dotenv()

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
    page_title="Kepler AI Data Agent",
    page_icon="🚗",
    layout="wide",
)

st.sidebar.title("分析エンジン設定")
model_option = st.sidebar.selectbox(
    "モデル選択",
    ["gemini-2.5-flash", "gemini-1.5-flash", "gpt-4o-mini", "gpt-4o"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### AI Data Analyst (PoC)
- **SQL Execution**: DuckDB による高速集計
- **Code Interpreter**: Python による可視化・統計解析
- **Semantic Layer**: カタログに基づく正確なデータ定義
- **Web Insights**: Gemini Native Search による最新情報補完
- **6-Layer Architecture**: システム工学に基づく高度な接地（Grounding）
""")

st.title("🚗 交通事故統計 AIデータエージェント")
st.caption("OpenAI Kepler アーキテクチャ準拠（6-Layer Grounded Context）")

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
                    status.write("📖 **Layer 1: Table Usage (Metadata) を確認中...**")
                elif name == "get_codex_enrichment":
                    status.write("🐍 **Layer 3: Codex Enrichment (前処理コード) を読解中...**")
                elif name == "get_learned_memory":
                    status.write("🧠 **Layer 5: Memory (過去の成功例・制約) を検索中...**")
                elif name == "run_runtime_context_query":
                    sql = ""
                    if isinstance(context, ToolContext):
                        try:
                            sql = json.loads(context.tool_arguments).get("sql", "")
                        except Exception:
                            pass
                    if sql:
                        status.write(f"🔍 **Layer 6: Runtime Context (ライブクエリ) 実行中:**\n```sql\n{sql}\n```")
                    else:
                        status.write("🔍 **Layer 6: Runtime Context 実行中...**")
                elif name == "execute_python":
                    status.write("📊 **Python 分析実行中 (Code Interpreter)...**")

            async def on_tool_end(self, context: Any, agent: Any, tool: Tool, result: str) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "run_runtime_context_query":
                    try:
                        rows = json.loads(result)
                        count = len(rows) if isinstance(rows, list) else "?"
                        status.write(f"✅ **DWH クエリ完了** — {count} 件取得")
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
                    status.write("🤖 **Kepler Data Agent に委譲してデータを集計中...**")
                elif name == "google_web_search":
                    query = ""
                    if isinstance(context, ToolContext):
                        try:
                            query = json.loads(context.tool_arguments).get("query", "")
                        except Exception:
                            pass
                    if query:
                        status.write(f"🌐 **Web Insights (Gemini Native Search) を実行中:**\n> {query}")
                    else:
                        status.write("🌐 **Web Insights を実行中...**")
                elif name == "get_institutional_knowledge":
                    status.write("📖 **Layer 4: Institutional Knowledge (社内知識) を確認中...**")

            async def on_tool_end(self, context: Any, agent: Any, tool: Tool, result: str) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "google_web_search":
                    status.write("✅ **Web Insights 完了** — 最新の社会情勢を取得しました")
                elif name == "query_data":
                    import re
                    plot_paths = re.findall(r"!\[.*?\]\((static/plots/.*?\.png)\)", result)
                    for p in plot_paths:
                        if p not in collected_plots:
                            collected_plots.append(p)
                            with plots_container:
                                st.image(p, caption=os.path.basename(p))
                            status.write(f"🖼️ **グラフ生成 (DataAgent)**: `{os.path.basename(p)}`")

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
