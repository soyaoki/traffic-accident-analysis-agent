import streamlit as st
import asyncio
import os
import json
import traceback
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
from agents.stream_events import RunItemStreamEvent
from agents.items import ToolCallItem, ToolCallOutputItem, MessageOutputItem

st.set_page_config(
    page_title="交通事故統計分析 AIエージェント",
    page_icon="🚗",
    layout="wide"
)

# サイドバーの設定
st.sidebar.title("設定")
model_option = st.sidebar.selectbox(
    "モデル選択",
    ["gemini-2.5-flash", "gemini-1.5-flash", "gpt-4o-mini", "gpt-4o"],
    index=0
)
with_context = st.sidebar.checkbox("コンテキスト（Layer 2 カタログ）を使用", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### このツールについて
警察庁の交通事故統計（原票番号単位）を、AIエージェントが多角的に分析します。
- **DataQueryAgent**: データの集計・抽出を担当
- **TrafficSafetyAnalyst**: 分析の解釈・統合を担当
""")

st.title("🚗 交通事故統計分析 AIエージェント")
st.markdown("""
2020年と2024年の交通事故データを比較・分析し、2030年目標（死者数半減）に向けたインサイトを抽出します。
""")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# サンプルクエリの提示
if not st.session_state.messages:
    st.markdown("### 試してみる質問例")
    cols = st.columns(3)
    sample_queries = [
        "2020年と2024年の死亡事故件数を比較して変化を教えて",
        "死亡負担が最も大きく、削減が停滞しているシナリオを特定して",
        "このままのペースで2030年目標を達成できますか？"
    ]
    for i, query in enumerate(sample_queries):
        if cols[i].button(query):
            st.session_state.prompt = query

# クエリ入力
prompt = st.chat_input("質問を入力してください（例：サポカーの効果はどうなっていますか？）")
if "prompt" in st.session_state and not prompt:
    prompt = st.session_state.pop("prompt")

if prompt:
    # ユーザーの入力を表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # エージェントの実行
    with st.chat_message("assistant"):
        status_placeholder = st.status("エージェントが分析を開始しました...", expanded=True)
        try:
            # nonlocal のスコープエラーを避けるため、ミュータブルなコンテナを使用
            response_container = {"full_content": ""}
            
            # 非同期実行
            async def process_stream():
                # エージェントのビルド
                _, analyst = build_agents(with_context=with_context, model=model_option)
                streaming_result = Runner.run_streamed(analyst, prompt)
                async for event in streaming_result.stream_events():
                    if isinstance(event, RunItemStreamEvent):
                        item = event.item
                        if isinstance(item, ToolCallItem):
                            agent_name = getattr(item.agent, 'name', 'Agent')
                            tool_name = item.tool_name
                            if tool_name == "query_traffic_data":
                                # 引数からクエリを抽出して表示
                                try:
                                    args_str = getattr(item.raw_item, 'arguments', '{}')
                                    args = json.loads(args_str)
                                    # Agent as tool usually uses 'input' or 'query'
                                    q = args.get("input") or args.get("query") or "データ集計"
                                    status_placeholder.write(f"🤖 **{agent_name}** がデータ取得を依頼: `{q}`")
                                except:
                                    status_placeholder.write(f"🤖 **{agent_name}** がデータ取得を依頼中...")
                            else:
                                status_placeholder.write(f"🛠️ **{agent_name}** がツール実行: `{tool_name}`")
                        elif isinstance(item, ToolCallOutputItem):
                            status_placeholder.write("📊 データの取得・集計が完了しました。")
                            # ToolCallOutputItem uses 'output'
                            tool_output = getattr(item, 'output', '')
                            with status_placeholder.expander("取得データの結果を確認"):
                                st.code(tool_output, language="json")
                        elif isinstance(item, MessageOutputItem):
                            status_placeholder.write("🧠 **TrafficSafetyAnalyst** が回答を生成しました。")
                
                # ストリーム終了後に最終出力を取得
                response_container["full_content"] = streaming_result.final_output
            
            asyncio.run(process_stream())
            full_response = response_container["full_content"]
            
            if full_response:
                status_placeholder.update(label="分析完了", state="complete", expanded=False)
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                status_placeholder.update(label="回答が空でした", state="error", expanded=True)
                st.warning("エージェントから有効な回答が得られませんでした。プロンプトを調整して再度お試しください。")
        except Exception as e:
            status_placeholder.update(label="エラー発生", state="error", expanded=True)
            st.error(f"分析中にエラーが発生しました: {e}")
            with st.expander("詳細なエラーログ"):
                st.code(traceback.format_exc())

    # レートリミット対策の案内（Gemini無料版などの場合）
    if "gemini" in model_option:
        st.info("💡 Gemini無料版をご利用の場合、連続した質問にはレート制限がかかる場合があります。")
