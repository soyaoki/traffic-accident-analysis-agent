"""データ層: CSVから DuckDB へのロードと接続管理。
DWH/DL 構成を想定し、全ての年度を単一の `accidents` テーブルとして扱う。
"""

import pandas as pd
import duckdb
from pathlib import Path
from functools import lru_cache

DATA_ROOT = Path(__file__).parent.parent / "data"
DB_PATH = DATA_ROOT / "traffic_safety.db"

# コードマップ（マスターデータとしても活用可能）
MASTER_MAPS = {
    "JIKO_RUIKEI": {1: "人対車両", 21: "車両相互", 41: "車両単独", 61: "列車"},
    "CHUU_YA": {
        11: "昼-明", 12: "昼-昼", 13: "昼-暮",
        21: "夜-暮", 22: "夜-夜", 23: "夜-明",
    },
    "TENKOU": {1: "晴", 2: "曇", 3: "雨", 4: "霧", 5: "雪"},
    "TIKEI": {1: "市街地-DID", 2: "市街地-その他", 3: "非市街地"},
    "ROAD_SHAPE": {
        1: "交さと点", 31: "環状交さと点", 7: "交さと点付近", 37: "環状交さと点付近",
        11: "トンネル", 12: "橋", 13: "カーブ", 14: "単路-その他",
        21: "踏切", 22: "踏切", 23: "踏切", 0: "一般交通の場所"
    },
    "SAPOCA": {0: "対象外", 1: "サポカー", 11: "非サポカー"},
    "INJURY": {0: "対象外", 1: "死亡", 2: "負傷", 4: "損傷なし"}
}

def _speed_band(code: int) -> str:
    if code in (1, 2, 3): return "低速(<=40)"
    if code in (4, 5): return "中速(<=60)"
    if code in (6, 7, 8, 9): return "高速(>60)"
    return "その他"

def get_connection():
    """DuckDB への接続を返す。"""
    return duckdb.connect(str(DB_PATH))

def init_db(force: bool = False):
    """
    データディレクトリから全ての CSV を DuckDB にインポートする。
    PoC段階では、実行時に DB がなければ作成する。
    """
    if DB_PATH.exists() and not force:
        return

    con = get_connection()
    all_dfs = []

    # 2020, 2024 等のフォルダをスキャン
    for year_dir in DATA_ROOT.glob("20*"):
        year = int(year_dir.name)
        csv_path = year_dir / f"honhyo_{year}.csv"
        if not csv_path.exists():
            continue
        
        print(f"Importing {year} data...")
        df = pd.read_csv(csv_path, encoding="shift_jis", low_memory=False)
        df["年"] = year
        
        # 共通のラベル列を追加 (Semantic Layer 的な前処理)
        df["事故類型名"] = df["事故類型"].map(MASTER_MAPS["JIKO_RUIKEI"]).fillna("不明")
        df["昼夜名"] = df["昼夜"].map(MASTER_MAPS["CHUU_YA"]).fillna("不明")
        df["夜間フラグ"] = df["昼夜"].isin([21, 22, 23])
        df["天候名"] = df["天候"].map(MASTER_MAPS["TENKOU"]).fillna("不明")
        df["地形名"] = df["地形"].map(MASTER_MAPS["TIKEI"]).fillna("不明")
        df["道路形状名"] = df["道路形状"].map(MASTER_MAPS["ROAD_SHAPE"]).fillna("不明")
        df["速度帯A"] = df["速度規制（指定のみ）（当事者A）"].apply(_speed_band)
        
        if "サポカー（当事者A）" in df.columns:
            df["サポカー名A"] = df["サポカー（当事者A）"].map(MASTER_MAPS["SAPOCA"]).fillna("不明")
        else:
            df["サポカー名A"] = "データなし"

        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # duckdb に登録
        con.execute("CREATE OR REPLACE TABLE accidents AS SELECT * FROM combined")
        print(f"Created 'accidents' table with {len(combined)} rows.")
    
    con.close()

# 互換性のための既存関数 (リファクタリング中に壊れないよう維持)
def load_honhyo(year: int) -> pd.DataFrame:
    con = get_connection()
    df = con.execute(f"SELECT * FROM accidents WHERE 年 = {year}").df()
    con.close()
    return df

def load_both_years() -> pd.DataFrame:
    con = get_connection()
    df = con.execute("SELECT * FROM accidents").df()
    con.close()
    return df
