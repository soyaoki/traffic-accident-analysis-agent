"""セマンティック・レイヤー、SQLクエリ、および Code Interpreter (Python実行)。
OpenAI Kepler アーキテクチャに準拠した6層の接地コンテキストを提供します。
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
    """[Layer 1: Table Usage] テーブルの利用実態と系統（Lineage）情報を取得する。
    過去の成功したクエリパターンや、通常どのテーブルがどう結合されるかの推論に使用する。"""
    if not _CATALOG_PATH.exists():
        return "Error: Catalog file not found."
    return _CATALOG_PATH.read_text(encoding="utf-8")

@function_tool
def get_human_annotations() -> str:
    """[Layer 2: Human Annotations] ドメインエキスパートによる厳選された説明を取得する。
    スキーマからは推測できない意図、セマンティクス、ビジネス上の意味、既知の注意事項を確認するために使用する。"""
    # 現在は catalog.yaml に集約
    return get_table_usage_metadata()

@function_tool
def get_codex_enrichment() -> str:
    """[Layer 3: Codex Enrichment] テーブルのコードレベルの定義を導き出す。
    preprocess.py を読み取り、値の一意性、データの更新頻度、除外されている粒度レベル（例：テストデータ除外等）などの
    微妙な差異をコードから抽出して理解するために使用する。"""
    if not _PREPROCESS_PATH.exists():
        return "Error: Source code not found."
    return _PREPROCESS_PATH.read_text(encoding="utf-8")

@function_tool
def get_institutional_knowledge() -> str:
    """[Layer 4: Institutional Knowledge] 組織内の知識（ドキュメント、リリース情報、インシデント等）を検索する。
    数値の変化が「バグ」なのか「施策」なのか「社会情勢」なのかを判断するための会社コンテキストを取得する。"""
    knowledge = ""
    if _DOMAIN_PATH.exists():
        knowledge += "--- Domain Knowledge ---\n" + _DOMAIN_PATH.read_text(encoding="utf-8") + "\n"
    if _BACKGROUND_PATH.exists():
        knowledge += "--- Background Info ---\n" + _BACKGROUND_PATH.read_text(encoding="utf-8") + "\n"
    return knowledge or "No institutional knowledge found."

@function_tool
def get_learned_memory() -> str:
    """[Layer 5: Memory] 過去に発見された微妙な差異、修正、フィルタリングの制約（例：特定の実験ゲートの文字列等）を取得する。
    他のレイヤーからは推測が難しい「わかりにくい修正」を再利用して、正確な回答のベースラインを確保するために使用する。"""
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
    """[Layer 5: Memory] 継続的な改善のため、今回の学習内容（フィルタ、制約、修正の微妙な差異）を保存する。
    複雑なフィルタリングを正しく行えた際に、次回同じ問題でつまずかないように必ず実行すること。"""
    con = get_connection()
    try:
        con.execute(
            "INSERT INTO query_learnings (topic, successful_sql, learning_note) VALUES (?, ?, ?)",
            [topic, sql, note]
        )
        return "Memory saved. This is now part of the global baseline for future responses."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        con.close()

@function_tool
def run_runtime_context_query(sql: str) -> str:
    """[Layer 6: Runtime Context] データウェアハウスにライブクエリを発行し、テーブルを直接調べたり検証したりする。
    事前コンテキストがない場合や古い場合に、スキーマを検証し、リアルタイムでデータを理解するために使用する。"""
    con = get_connection()
    try:
        df = con.execute(sql).df()
        if len(df) > 100:
            return df.head(100).to_json(orient="records", force_ascii=False)
        return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        # Runtime Feedback
        tables = con.execute("SHOW TABLES").df()
        return f"Runtime feedback: {str(e)}\nAvailable tables: {tables['name'].tolist()}"
    finally:
        con.close()

@function_tool
def execute_python(code: str) -> str:
    """Pythonによる可視化・統計解析を実行する。"""
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
    """Gemini Native Search による最新情報（2024年以降の法改正等）の取得。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return "Error: API key missing."
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    try:
        response = client.models.generate_content(
            model=os.getenv("AGENT_MODEL", "gemini-2.5-flash"),
            contents=query,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        )
        return response.text
    except Exception as e: return f"Search error: {str(e)}"

DATA_TOOLS = [
    get_table_usage_metadata,
    get_human_annotations,
    get_codex_enrichment,
    get_learned_memory,
    save_memory,
    run_runtime_context_query,
    execute_python,
]

ANALYST_TOOLS = [
    get_institutional_knowledge,
    google_web_search,
]
