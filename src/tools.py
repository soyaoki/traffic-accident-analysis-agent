"""セマンティック・レイヤー、SQLクエリ、および Code Interpreter (Python実行)。
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
def get_semantic_catalog() -> str:
    """[Layer 2: Annotations] データカタログの定義を返す。指標やカラム定義、利用パターンの確認に使用する。"""
    if not _CATALOG_PATH.exists():
        return "Error: Catalog file not found."
    return _CATALOG_PATH.read_text(encoding="utf-8")

@function_tool
def get_data_lineage_code() -> str:
    """[Layer 3: Code-derived Context] データの生成ロジック（preprocess.py）を読み取る。
    カラムの変換ルール、マスター値の定義、ETLの仕様を確認し、データの正確な由来を把握するために使用する。"""
    if not _PREPROCESS_PATH.exists():
        return "Error: Source code not found."
    return _PREPROCESS_PATH.read_text(encoding="utf-8")

@function_tool
def get_learned_memory() -> str:
    """[Layer 5: Learned Memory] 過去の成功した複雑なクエリや、失敗から学んだ「修正済みの知識」を取得する。
    同じミスを防ぎ、より洗練されたクエリを作成するために使用する。"""
    con = get_connection()
    try:
        # テーブルがなければ作成（Layer 5 の初期化）
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
            return "No previous learnings found. You are exploring this topic for the first time."
        return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        return f"Error retrieving learnings: {str(e)}"
    finally:
        con.close()

@function_tool
def save_query_learning(topic: str, sql: str, note: str) -> str:
    """[Layer 5: Learned Memory] 成功したクエリや分析の学びを保存する。
    複雑な条件のSQLが正しく動いた際や、エラーを修正した際に必ず実行すること。"""
    con = get_connection()
    try:
        con.execute(
            "INSERT INTO query_learnings (topic, successful_sql, learning_note) VALUES (?, ?, ?)",
            [topic, sql, note]
        )
        return "Learning saved successfully. This will improve future performance."
    except Exception as e:
        return f"Error saving learning: {str(e)}"
    finally:
        con.close()

@function_tool
def get_domain_knowledge() -> str:
    """[Layer 3: Institutional Knowledge] 交通安全統計のドメイン知識（バイアス・分析方針・解釈指針）を返す。
    分析結果を解釈・説明する前に必ず参照すること。"""
    if not _DOMAIN_PATH.exists():
        return "Error: Domain knowledge file not found."
    return _DOMAIN_PATH.read_text(encoding="utf-8")

@function_tool
def get_background_knowledge() -> str:
    """[Layer 4: Institutional Knowledge] 社会情勢や法改正などの背景知識（外的要因・データの収集背景）を返す。
    データ上の変化が、社会的な出来事（コロナ、法改正等）とどう関連しているか考察する際に参照すること。"""
    if not _BACKGROUND_PATH.exists():
        return "Error: Background knowledge file not found."
    return _BACKGROUND_PATH.read_text(encoding="utf-8")

@function_tool
def run_traffic_query(sql: str) -> str:
    """
    [Layer 1 & 6: Schema & Runtime Validation] 
    accidentsテーブルに対してSQLを実行し、結果を返す。
    エラーが発生した場合は、そのフィードバックを元に自己修正（Runtime Correction）を試みること。
    """
    con = get_connection()
    try:
        df = con.execute(sql).df()
        if len(df) > 100:
            return df.head(100).to_json(orient="records", force_ascii=False)
        return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        # [Layer 6] 実行時のエラーフィードバック
        tables = con.execute("SHOW TABLES").df()
        table_list = tables['name'].tolist() if not tables.empty else []
        error_msg = f"Runtime Error: {str(e)}\n"
        error_msg += f"Available tables: {table_list}. Please check your SQL and fix it based on this feedback."
        return error_msg
    finally:
        con.close()

@function_tool
def execute_python(code: str) -> str:
    """
    Pythonコードを実行し、標準出力と生成されたグラフの情報を返す。
    高度な統計解析、比較、可視化に使用すること。
    
    環境には以下の変数がプリセットされている:
    - `pd`: pandas
    - `plt`: matplotlib.pyplot
    - `sns`: seaborn
    - `db_path`: DuckDBファイルのパス
    
    注意:
    - グラフを作成した場合は `plt.savefig('static/plots/filename.png')` のように保存すること。
    - 保存したファイル名は出力に含めること。
    """
    import japanize_matplotlib  # noqa: F401
    plt.switch_backend("Agg")

    loc = {
        "pd": pd,
        "plt": plt,
        "sns": sns,
        "db_path": str(DB_PATH),
        "duckdb": duckdb,
    }

    # sys.stdout をコンテキストマネージャで差し替え（例外でも確実に戻す）
    output = io.StringIO()
    before_plots = set(_PLOT_DIR.glob("*.png"))

    try:
        with contextlib.redirect_stdout(output):
            exec(code, globals(), loc)  # noqa: S102

        # 絶対パスからプロジェクトルートからの相対パス (static/plots/...) に変換して返す
        new_plots = [str(p.relative_to(Path.cwd())) for p in set(_PLOT_DIR.glob("*.png")) - before_plots]
        return json.dumps(
            {"stdout": output.getvalue(), "message": "Execution successful", "plots": new_plots},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"error": str(e), "stdout": output.getvalue()},
            ensure_ascii=False,
        )

@function_tool
def google_web_search(query: str) -> str:
    """
    Google検索（Gemini Native Web Search）を使用して、最新のニュースや統計情報を取得する。
    
    注意:
    - 2024年以降の出来事や、データベースに含まれない最新の背景知識を補完するために使用すること。
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is not set."

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    model_id = os.getenv("AGENT_MODEL", "gemini-2.5-flash")
    
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        return response.text
    except Exception as e:
        return f"Error during Google Search: {str(e)}"

DATA_TOOLS = [
    get_semantic_catalog,
    get_data_lineage_code,
    get_learned_memory,
    save_query_learning,
    run_traffic_query,
    execute_python,
]

ANALYST_TOOLS = [
    get_domain_knowledge,
    get_background_knowledge,
    google_web_search,
]
