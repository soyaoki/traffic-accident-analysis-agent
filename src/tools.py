"""セマンティック・レイヤー、SQLクエリ、および Code Interpreter (Python実行)。
"""

import json
import io
import sys
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from agents import function_tool
from src.preprocess import get_connection, DB_PATH

_CATALOG_PATH = Path(__file__).parent / "context" / "catalog.yaml"
_PLOT_DIR = Path(__file__).parent.parent / "static" / "plots"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)

@function_tool
def get_semantic_catalog() -> str:
    """データカタログの定義を返す。SQLを書く前に必ず確認すること。"""
    if not _CATALOG_PATH.exists():
        return "Error: Catalog file not found."
    return _CATALOG_PATH.read_text(encoding="utf-8")

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
    # 出力をキャプチャ
    output = io.StringIO()
    sys.stdout = output
    
    # 実行コンテキスト
    loc = {
        "pd": pd,
        "plt": plt,
        "sns": sns,
        "db_path": str(DB_PATH),
        "duckdb": duckdb
    }
    
    try:
        # matplotlib の非インタラクティブバックエンド設定
        plt.switch_backend('Agg')
        # Python 3.12+ で削除された distutils 対策として setuptools を明示的にロード
        import setuptools 
        import japanize_matplotlib
        
        # 実行前のファイルリストを取得
        before_plots = set(_PLOT_DIR.glob("*.png"))
        
        exec(code, globals(), loc)
        sys.stdout = sys.__stdout__
        result_text = output.getvalue()
        
        # 実行後に増えたファイル（＝今回作成されたグラフ）のみを特定
        after_plots = set(_PLOT_DIR.glob("*.png"))
        new_plots = [str(p) for p in (after_plots - before_plots)]
        
        return json.dumps({
            "stdout": result_text,
            "message": "Execution successful",
            "plots": new_plots
        }, ensure_ascii=False)
        
    except Exception as e:
        sys.stdout = sys.__stdout__
        return json.dumps({
            "error": str(e),
            "stdout": output.getvalue()
        }, ensure_ascii=False)

ALL_TOOLS = [
    get_semantic_catalog,
    run_traffic_query,
    execute_python,
]
