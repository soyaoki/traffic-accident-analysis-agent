import streamlit as st
import asyncio
import os
import json
import traceback
from pathlib import Path
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# Gemini使用時やOpenAIキーがプレースホルダの場合は、agentsのトレース送信を最優先で無効化
from agents.tracing import set_tracing_disabled
_has_openai_key = bool(os.getenv("OPENAI_API_KEY", "").startswith("sk-")) and len(os.getenv("OPENAI_API_KEY", "")) > 10
if not _has_openai_key:
    set_tracing_disabled(True)

from agents import Runner
from src.agent import build_agents
from agents.items import ToolCallItem, ToolCallOutputItem, MessageOutputItem

st.set_page_config(
    page_title="交通事故統計分析 AIエージェント (OpenAI Style)",
    page_icon="🚗",
    layout="wide"
)

# サイドバーの設定
st.sidebar.title("分析エンジン設定")
model_option = st.sidebar.selectbox(
    "モデル選択",
    ["gemini-2.5-flash", "gemini-1.5-flash", "gpt-4o-mini", "gpt-4o"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### AI Data Analyst (PoC)
- **SQL Execution**: DuckDB による高速集計
- **Code Interpreter**: Python による可視化・統計解析
- **Semantic Layer**: カタログに基づく正確な意味理解
""")

st.title("🚗 交通事故統計 AIデータエージェント")
st.caption("SQL + Code Interpreter を活用した高度な分析デモ")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "plots" in message:
            for plot_path in message["plots"]:
                if os.path.exists(plot_path):
                    st.image(plot_path)

# サンプルクエリの提示
if not st.session_state.messages:
    st.markdown("### 試してみる質問例")
    cols = st.columns(3)
    sample_queries = [
        "2020年と2024年の死亡事故件数の比較をグラフにして",
        "人対車両の事故において、速度帯と死亡率の関係を分析して",
        "このままのペースで2030年目標を達成可能か試算して"
    ]
    for i, query in enumerate(sample_queries):
        if cols[i].button(query):
            st.session_state.prompt = query

# クエリ入力
prompt = st.chat_input("質問を入力してください（例：2020年と2024年の変化をグラフにして）")
if "prompt" in st.session_state and not prompt:
    prompt = st.session_state.pop("prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # グラフ表示用の領域を確保（テキストの上に表示）
        plot_placeholder = st.container()
        response_placeholder = st.empty()
        status_placeholder = st.status("エージェントが分析中...", expanded=True)
        
        try:
            response_container = {"full_content": "", "plots": []}
            
            async def process_stream():
                # エージェントのビルド
                _, analyst = build_agents(model=model_option)
                
                streaming_result = Runner.run_streamed(analyst, prompt)
                async for event in streaming_result.stream_events():
                    # 1. アイテム単位の処理
                    if event.type == "run_item_stream_event":
                        item = event.item
                        
                        # ツール呼び出しの表示
                        if isinstance(item, ToolCallItem):
                            tool_name = item.tool_name
                            if tool_name == "run_traffic_query":
                                try:
                                    args = json.loads(item.raw_item.arguments)
                                    sql = args.get("sql", "")
                                    status_placeholder.write(f"🔍 **SQL実行:**\n```sql\n{sql}\n```")
                                except:
                                    status_placeholder.write("🔍 **SQL実行中...**")
                            elif tool_name == "execute_python":
                                status_placeholder.write("🐍 **Python実行中 (Code Interpreter)...**")
                            elif tool_name == "get_semantic_catalog":
                                status_placeholder.write("📖 **カタログ定義を確認中...**")
                        
                        # ツール結果の表示
                        elif isinstance(item, ToolCallOutputItem):
                            try:
                                out_data = json.loads(item.output)
                                if "plots" in out_data:
                                    for p in out_data["plots"]:
                                        if p not in response_container["plots"]:
                                            response_container["plots"].append(p)
                                            # 専用コンテナに画像を表示
                                            with plot_placeholder:
                                                st.image(p, caption=f"Generated Plot: {os.path.basename(p)}")
                                            status_placeholder.write(f"🖼️ グラフを生成しました: `{os.path.basename(p)}`")
                                
                                with status_placeholder.expander(f"ツール実行結果: {getattr(item, 'tool_name', 'Output')}"):
                                    st.json(out_data)
                            except:
                                pass

                    # 2. リアルタイムのテキスト生成（もし利用可能な場合）
                    elif event.type == "run_item_delta_event":
                        if hasattr(event, 'delta') and event.delta.type == "text_delta":
                            response_container["full_content"] += event.delta.text
                            response_placeholder.markdown(response_container["full_content"] + "▌")

                # 最終的な確定テキストを表示
                final_output = streaming_result.final_output
                response_placeholder.markdown(final_output)
                response_container["full_content"] = final_output
                return response_container

            res_data = asyncio.run(process_stream())
            
            if res_data["full_content"]:
                status_placeholder.update(label="分析完了", state="complete", expanded=False)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": res_data["full_content"],
                    "plots": res_data["plots"]
                })
            else:
                status_placeholder.update(label="完了しましたが回答がありません", state="complete")

        except Exception as e:
            status_placeholder.update(label="エラー発生", state="error", expanded=True)
            st.error(f"エラー: {e}")
            st.code(traceback.format_exc())
