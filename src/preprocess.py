"""データ読み込みと前処理。Shift-JIS CSV → ラベル付き DataFrame。

2020(58列) vs 2024(68列) の差異に注意：
  2024にあって2020にない主要列:
    サポカー（当事者A/B）, オートマチック車（当事者A/B）,
    認知機能検査経過日数（当事者A/B）, 運転練習の方法（当事者A/B）,
    日の出/日の入り時刻（時/分）
  2020にあって2024にない:
    車両形状（当事者A/B）→ 2024では「車両形状等（当事者A/B）」に改称,
    環状交差点の直径, 上下線
"""

import pandas as pd
from pathlib import Path
from functools import lru_cache

DATA_ROOT = Path(__file__).parent.parent / "data"

# コードマップ（catalog.yaml と対応）
JIKO_RUIKEI = {1: "人対車両", 21: "車両相互", 41: "車両単独", 61: "列車"}
CHUU_YA = {
    11: "昼-明", 12: "昼-昼", 13: "昼-暮",
    21: "夜-暮", 22: "夜-夜", 23: "夜-明",
}
NIGHT_CODES = {21, 22, 23}
SAPOCA = {0: "対象外", 1: "サポカー", 11: "非サポカー"}  # 11=非サポカー(~91%の死亡事故), 1=サポカー
INJURY = {0: "対象外", 1: "死亡", 2: "負傷", 4: "損傷なし"}
TENKOU = {1: "晴", 2: "曇", 3: "雨", 4: "霧", 5: "雪"}
TIKEI = {1: "市街地-DID", 2: "市街地-その他", 3: "非市街地"}
ROAD_SHAPE = {
    1: "交差点", 31: "環状交差点", 7: "交差点付近", 37: "環状交差点付近",
    11: "トンネル", 12: "橋", 13: "カーブ", 14: "単路-その他",
    21: "踏切", 22: "踏切", 23: "踏切", 0: "一般交通の場所"
}


def _speed_band(code: int) -> str:
    if code in (1, 2, 3):
        return "低速(≤40)"
    if code in (4, 5):
        return "中速(≤60)"
    if code in (6, 7, 8, 9):
        return "高速(>60)"
    return "その他"


@lru_cache(maxsize=4)
def load_honhyo(year: int) -> pd.DataFrame:
    """本票を読み込み、コードをラベルに変換して返す。"""
    path = DATA_ROOT / str(year) / f"honhyo_{year}.csv"
    df = pd.read_csv(path, encoding="shift_jis", low_memory=False)

    df["事故類型名"] = df["事故類型"].map(JIKO_RUIKEI).fillna("不明")
    df["昼夜名"] = df["昼夜"].map(CHUU_YA).fillna("不明")
    df["夜間フラグ"] = df["昼夜"].isin(NIGHT_CODES)
    df["天候名"] = df["天候"].map(TENKOU).fillna("不明")
    df["地形名"] = df["地形"].map(TIKEI).fillna("不明")
    df["道路形状名"] = df["道路形状"].map(ROAD_SHAPE).fillna("不明")
    df["速度帯A"] = df["速度規制（指定のみ）（当事者A）"].apply(_speed_band)
    df["損傷程度A"] = df["人身損傷程度（当事者A）"].map(INJURY).fillna("不明")
    df["損傷程度B"] = df["人身損傷程度（当事者B）"].map(INJURY).fillna("不明")
    df["年"] = year

    # サポカー列は2024のみ存在（2020にはない）
    if "サポカー（当事者A）" in df.columns:
        df["サポカー名A"] = df["サポカー（当事者A）"].map(SAPOCA).fillna("不明")
    else:
        df["サポカー名A"] = "データなし（2020年は非収録）"

    return df


def load_both_years() -> pd.DataFrame:
    """2020・2024の本票を結合して返す。列の差異は NaN で埋める。"""
    return pd.concat([load_honhyo(2020), load_honhyo(2024)], ignore_index=True)
