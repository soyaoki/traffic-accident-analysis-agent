"""セマンティック・レイヤー、SQLクエリ、および Code Interpreter (Python実行)。
"""

import contextlib
import json
import io
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from agents import function_tool, WebSearchTool
from src.preprocess import get_connection, DB_PATH

_CATALOG_PATH = Path(__file__).parent / "context" / "catalog.yaml"
_DOMAIN_PATH = Path(__file__).parent / "context" / "domain.yaml"
_BACKGROUND_PATH = Path(__file__).parent / "context" / "background.yaml"
_PLOT_DIR = Path(__file__).parent.parent / "static" / "plots"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)

@function_tool
def get_semantic_catalog() -> str:
    """データカタログの定義を返す。SQLを書く前に必ず確認すること。"""
    if not _CATALOG_PATH.exists():
        return "Error: Catalog file not found."
    return _CATALOG_PATH.read_text(encoding="utf-8")

@function_tool
def get_domain_knowledge() -> str:
    """交通安全統計のドメイン知識（バイアス・分析方針・解釈指針）を返す。
    分析結果を解釈・説明する前に必ず参照すること。"""
    if not _DOMAIN_PATH.exists():
        return "Error: Domain knowledge file not found."
    return _DOMAIN_PATH.read_text(encoding="utf-8")

@function_tool
def get_background_knowledge() -> str:
    """社会情勢や法改正などの背景知識（外的要因・データの収集背景）を返す。
    データ上の変化が、社会的な出来事（コロナ、法改正等）とどう関連しているか考察する際に参照すること。"""
    if not _BACKGROUND_PATH.exists():
        return "Error: Background knowledge file not found."
    return _BACKGROUND_PATH.read_text(encoding="utf-8")

@function_tool
def run_traffic_query(sql: str) -> str:
    """
    accidentsテーブルに対してSQLを実行し、結果をJSONで返す。
    
    注意:
    - 全てのデータ（2020年, 2024年）はすでに `accidents` テーブルにロードされています。
    - `READ_CSV` などを使ってファイルを直接読み込もうとせず、必ず `FROM accidents` を使用してください。
    """
    con = get_connection()
    try:
        df = con.execute(sql).df()
        if len(df) > 100:
            return df.head(100).to_json(orient="records", force_ascii=False)
        return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        # エラー時に利用可能なテーブルを表示して、迷走（CSV直接読み込みの試行）を防ぐ
        tables = con.execute("SHOW TABLES").df()
        table_list = tables['name'].tolist() if not tables.empty else []
        return f"Error: {str(e)}\nHint: Available tables are {table_list}. Please query 'accidents' table."
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

        new_plots = [str(p) for p in set(_PLOT_DIR.glob("*.png")) - before_plots]
        return json.dumps(
            {"stdout": output.getvalue(), "message": "Execution successful", "plots": new_plots},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"error": str(e), "stdout": output.getvalue()},
            ensure_ascii=False,
        )

DATA_TOOLS = [
    get_semantic_catalog,
    run_traffic_query,
    execute_python,
]

ANALYST_TOOLS = [
    get_domain_knowledge,
    get_background_knowledge,
    WebSearchTool(),
]
