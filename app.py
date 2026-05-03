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
        "2020年比で2030年の「死亡事故半減」は達成できそう？未達見込みならどの事故形態にどう手を打つべきか、データと最新の法改正を踏まえて戦略を練って",
        "2024年の自転車事故や死亡率に変化はある？ヘルメット着用努力義務化などの影響も踏まえて分析して",
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

        class DataEngineerHooks(AgentHooksBase):
            async def on_tool_start(self, context: Any, agent: Any, tool: Tool) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "get_table_usage_metadata":
                    status.write("🛠️ **DataEngineer**: Layer 1 (Usage) を接地中...")
                elif name == "get_codex_enrichment":
                    status.write("🛠️ **DataEngineer**: Layer 3 (Code) を接地中...")
                elif name == "get_learned_memory":
                    status.write("🛠️ **DataEngineer**: Layer 5 (Memory) を接地中...")
                elif name == "run_runtime_context_query":
                    status.write("🛠️ **DataEngineer**: Layer 6 (Runtime) クエリ実行中...")

            async def on_tool_end(self, context: Any, agent: Any, tool: Tool, result: str) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "run_runtime_context_query":
                    status.write("✅ **DataEngineer**: データの取得が完了しました。")

        class DataAnalystHooks(AgentHooksBase):
            async def on_tool_start(self, context: Any, agent: Any, tool: Tool) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "get_human_annotations":
                    status.write("📊 **DataAnalyst**: Layer 2 (Annotations) を接地中...")
                elif name == "request_data_retrieval":
                    status.write("📊 **DataAnalyst**: エンジニアにデータ取得を依頼中...")
                elif name == "execute_python":
                    status.write("📊 **DataAnalyst**: Pythonで可視化・分析を実行中...")
                elif name == "get_institutional_knowledge":
                    status.write("📊 **DataAnalyst**: Layer 4 (Institutional) を接地中...")

            async def on_tool_end(self, context: Any, agent: Any, tool: Tool, result: str) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "request_data_retrieval":
                    with status:
                        with st.expander("取得データ（Raw）", expanded=False):
                            st.code(result[:1000] + ("..." if len(result) > 1000 else ""), language="json")
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

        class ManagerHooks(AgentHooksBase):
            async def on_tool_start(self, context: Any, agent: Any, tool: Tool) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "request_analysis":
                    status.write("🛡️ **Manager**: アナリストに分析を指示中...")
                elif name == "google_web_search":
                    status.write("🌐 **Manager**: 外部情報を調査中...")

            async def on_tool_end(self, context: Any, agent: Any, tool: Tool, result: str) -> None:
                name = tool.name if hasattr(tool, "name") else str(tool)
                if name == "request_analysis":
                    status.write("✅ **Manager**: アナリストからの報告を受領しました。")
                elif name == "google_web_search":
                    with status:
                        with st.expander("Web検索結果の要約", expanded=False):
                            st.write(result)

        try:
            async def run_agent():
                _, _, _, manager = build_agents(
                    model=model_option,
                    engineer_hooks=DataEngineerHooks(),
                    analyst_hooks=DataAnalystHooks(),
                    manager_hooks=ManagerHooks(),
                )
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                return await Runner.run(manager, history)

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
