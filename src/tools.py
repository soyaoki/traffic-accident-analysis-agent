"""セマンティック・レイヤー、SQLクエリ、および Code Interpreter (Python実行)。
OpenAI が公開した「6層の接地されたコンテキスト (6 layers of grounded context)」に基づき構築。
"""

import os
import contextlib
import json
import io
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from google import genai
from google.genai import types
from agents import function_tool
from src.preprocess import get_connection, DB_PATH

_CATALOG_PATH = Path(__file__).parent / "context" / "catalog.yaml"
_DOMAIN_PATH = Path(__file__).parent / "context" / "domain.yaml"
_BACKGROUND_PATH = Path(__file__).parent / "context" / "background.yaml"
_PREPROCESS_PATH = Path(__file__).parent / "preprocess.py"
_PLOT_DIR = Path(__file__).parent.parent / "static" / "plots"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)

@function_tool
def get_table_usage_metadata() -> str:
    """[レイヤー #1：テーブルの使用状況] スキーマメタデータ（列名とデータ型）およびテーブル系統（Lineage）を取得する。
    過去のクエリを取り込むことで、クエリの記述方法や、通常どのテーブルが結合されるかの理解に使用する。"""
    if not _CATALOG_PATH.exists():
        return "Error: Catalog file not found."
    return _CATALOG_PATH.read_text(encoding="utf-8")

@function_tool
def get_human_annotations() -> str:
    """[レイヤー #2：人間による注釈] 指標（致死率等）の計算定義、ビジネスルール、分析上の注意事項を取得する。
    いかなる分析・計算を開始する前にも、必ず最初にこのツールを呼び出して定義を接地させること。"""
    if not _CATALOG_PATH.exists():
        return "Error: Catalog file not found."
    return _CATALOG_PATH.read_text(encoding="utf-8")

@function_tool
def get_codex_enrichment() -> str:
    """[レイヤー #3：Codex エンリッチメント] テーブルのコードレベルの定義を導き出す。
    ソースコードから値の一意性、データの更新頻度、データの範囲（除外されている粒度レベルなど）を理解するために使用する。"""
    if not _PREPROCESS_PATH.exists():
        return "Error: Source code not found."
    return _PREPROCESS_PATH.read_text(encoding="utf-8")

@function_tool
def get_institutional_knowledge() -> str:
    """[レイヤー #4：インスティテューショナルナレッジ] 社内のドキュメント（Slack、Notion等）に相当するコンテキストを取得する。
    リリース、信頼性インシデント、主要メトリックの標準定義と計算ロジックなどの重要な会社のコンテキストを確認するために使用する。"""
    knowledge = ""
    if _DOMAIN_PATH.exists():
        knowledge += "--- Domain Knowledge ---\n" + _DOMAIN_PATH.read_text(encoding="utf-8") + "\n"
    if _BACKGROUND_PATH.exists():
        knowledge += "--- Background Info ---\n" + _BACKGROUND_PATH.read_text(encoding="utf-8") + "\n"
    return knowledge or "No institutional knowledge found."

@function_tool
def get_learned_memory() -> str:
    """[レイヤー #5：メモリ] 過去の修正、フィルタ、制約など、他のレイヤーから推測が難しい「わかりにくい修正」を取得する。
    同じ問題でつまずかないよう、学習内容を次回の回答の正確なベースラインとして再利用するために使用する。"""
    con = get_connection()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS query_learnings (
                id INTEGER PRIMARY KEY,
                topic TEXT,
                successful_sql TEXT,
                learning_note TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        df = con.execute("SELECT topic, successful_sql, learning_note FROM query_learnings ORDER BY created_at DESC LIMIT 5").df()
        if df.empty:
            return "No memory found. You must establish a new baseline for this query."
        return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        con.close()

@function_tool
def save_memory(topic: str, sql: str, note: str) -> str:
    """[レイヤー #5：メモリ] 特定のデータに関する修正や学習内容を次回のために保存する。
    ユーザーによる修正や、他のレイヤーからは推論しづらい制約を発見した際に必ず実行すること。"""
    con = get_connection()
    try:
        con.execute(
            "INSERT INTO query_learnings (topic, successful_sql, learning_note) VALUES (?, ?, ?)",
            [topic, sql, note]
        )
        return "Memory saved. Future responses will be based on this more accurate baseline."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        con.close()

@function_tool
def run_runtime_context_query(sql: str) -> str:
    """[レイヤー #6：ランタイムコンテキスト] データウェアハウスにライブクエリを発行して、テーブルを直接調べたりクエリしたりする。
    スキーマを検証し、データをリアルタイムで理解するために使用する。"""
    con = get_connection()
    try:
        df = con.execute(sql).df()
        if len(df) > 100:
            return df.head(100).to_json(orient="records", force_ascii=False)
        return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        tables = con.execute("SHOW TABLES").df()
        return f"Runtime error: {str(e)}\nAvailable tables: {tables['name'].tolist()}"
    finally:
        con.close()

@function_tool
def execute_python(code: str) -> str:
    """Pythonによる分析・可視化を実行する。"""
    import japanize_matplotlib  # noqa: F401
    plt.switch_backend("Agg")
    loc = {"pd": pd, "plt": plt, "sns": sns, "db_path": str(DB_PATH), "duckdb": duckdb}
    output = io.StringIO()
    before_plots = set(_PLOT_DIR.glob("*.png"))
    try:
        with contextlib.redirect_stdout(output):
            exec(code, globals(), loc)  # noqa: S102
        new_plots = [str(p.relative_to(Path.cwd())) for p in set(_PLOT_DIR.glob("*.png")) - before_plots]
        return json.dumps({"stdout": output.getvalue(), "message": "Success", "plots": new_plots}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "stdout": output.getvalue()}, ensure_ascii=False)

@function_tool
def google_web_search(query: str) -> str:
    """最新情報（法改正等）をGoogle検索で取得する。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return "Error: API key missing."
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    try:
        response = client.models.generate_content(
            model=os.getenv("AGENT_MODEL", "gemini-2.5-flash"),
            contents=query,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        )
        
        # 参照サイト名の抽出（リンクは404になるため名前のみ）
        source_names = []
        try:
            if hasattr(response, "candidates") and response.candidates:
                metadata = response.candidates[0].grounding_metadata
                if metadata and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        if chunk.web and chunk.web.title:
                            source_names.append(chunk.web.title)
        except Exception:
            pass

        import urllib.parse
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        
        result_text = response.text
        if source_names:
            unique_sources = ", ".join(list(dict.fromkeys(source_names)))
            result_text += f"\n\n### 参照情報（External Sources）\n- **主な情報源**: {unique_sources}"
        
        result_text += f"\n- [Google 検索で詳細を確認する]({search_url})"
            
        return result_text
    except Exception as e: return f"Search error: {str(e)}"

# ── Role-based Tool Groups ────────────────────────────────────────────────

ENGINEER_TOOLS = [
    get_table_usage_metadata,  # Layer 1
    get_codex_enrichment,      # Layer 3
    get_learned_memory,        # Layer 5
    save_memory,               # Layer 5
    run_runtime_context_query,  # Layer 6
]

ANALYST_TOOLS = [
    get_human_annotations,       # Layer 2
    get_institutional_knowledge,  # Layer 4
    execute_python,              # Analysis & Visualization
]

MANAGER_TOOLS = [
    google_web_search,           # External Insights
]

SCIENTIST_TOOLS = []  # Placeholder for future DataScientist tools
