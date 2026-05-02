"""OAI Agents SDK の function_tool として登録する分析ツール群。

指標の定義:
  死亡事故件数 = (死者数 > 0).sum()  ← 参考記事の数値と一致（2020=2784, 2024=2598）
  死者数合計   = 死者数.sum()        ← 1事故で複数死者の場合に上回る
  致死率       = 死亡事故件数 / 全事故件数
"""

import json
import pandas as pd
from agents import function_tool
from src.preprocess import load_honhyo, load_both_years


@function_tool
def get_overview(year: int) -> str:
    """
    指定年の交通事故統計の基本サマリーを返す。
    year: 2020 または 2024
    """
    df = load_honhyo(year)
    total = len(df)
    fatal_accidents = int((df["死者数"] > 0).sum())
    fatality_rate = fatal_accidents / total * 100

    by_type = (
        df.groupby("事故類型名")
        .agg(
            全事故件数=("死者数", "count"),
            死亡事故件数=("死者数", lambda x: (x > 0).sum()),
        )
        .assign(致死率=lambda x: (x["死亡事故件数"] / x["全事故件数"] * 100).round(3))
    )

    return json.dumps(
        {
            "year": year,
            "全事故件数": total,
            "死亡事故件数": fatal_accidents,
            "致死率(%)": round(fatality_rate, 3),
            "事故類型別": by_type.to_dict(),
        },
        ensure_ascii=False,
        indent=2,
    )


@function_tool
def compare_years() -> str:
    """
    2020年と2024年の主要指標を比較する。
    死亡事故件数・全事故件数・致死率の変化率と絶対値差を返す。
    """
    d20 = load_honhyo(2020)
    d24 = load_honhyo(2024)

    def stats(df: pd.DataFrame) -> dict:
        total = len(df)
        fatal = int((df["死者数"] > 0).sum())
        return {"全事故件数": total, "死亡事故件数": fatal, "致死率(%)": round(fatal / total * 100, 3)}

    s20, s24 = stats(d20), stats(d24)
    annual_trend = (s24["死亡事故件数"] - s20["死亡事故件数"]) / 4

    return json.dumps(
        {
            "2020": s20,
            "2024": s24,
            "変化": {
                "死亡事故件数差": s24["死亡事故件数"] - s20["死亡事故件数"],
                "死亡事故件数変化率(%)": round(
                    (s24["死亡事故件数"] - s20["死亡事故件数"]) / s20["死亡事故件数"] * 100, 1
                ),
                "全事故件数変化率(%)": round(
                    (s24["全事故件数"] - s20["全事故件数"]) / s20["全事故件数"] * 100, 1
                ),
                "年間トレンド(死亡事故件数/年)": round(annual_trend, 1),
            },
        },
        ensure_ascii=False,
        indent=2,
    )


@function_tool
def analyze_fatal_scenarios() -> str:
    """
    事故類型 × 昼夜（昼間/夜間） × 速度帯 で死亡事故をセグメント分析する。
    2020→2024の変化と死亡負担（死亡事故件数絶対値）を返す。
    ADASの効果が薄い「介入空白」セグメントの特定に使う。
    """
    df = load_both_years()
    df["昼夜区分"] = df["夜間フラグ"].map({True: "夜間", False: "昼間"})
    df["死亡フラグ"] = (df["死者数"] > 0).astype(int)

    grp = (
        df.groupby(["年", "事故類型名", "昼夜区分", "速度帯A"])
        .agg(死亡事故件数=("死亡フラグ", "sum"))
        .reset_index()
    )

    pivot = grp.pivot_table(
        index=["事故類型名", "昼夜区分", "速度帯A"],
        columns="年",
        values="死亡事故件数",
        fill_value=0,
    )
    pivot.columns = [f"死亡事故件数_{int(y)}" for y in pivot.columns]
    pivot = pivot.reset_index()
    pivot["変化"] = pivot["死亡事故件数_2024"] - pivot["死亡事故件数_2020"]
    pivot["変化率(%)"] = (
        (pivot["死亡事故件数_2024"] - pivot["死亡事故件数_2020"])
        / pivot["死亡事故件数_2020"].replace(0, float("nan"))
        * 100
    ).round(1)

    top = pivot.nlargest(15, "死亡事故件数_2024")[
        ["事故類型名", "昼夜区分", "速度帯A",
         "死亡事故件数_2020", "死亡事故件数_2024", "変化", "変化率(%)"]
    ]

    return json.dumps(
        {
            "説明": "死亡事故件数が多いセグメント上位15。変化率がプラスまたはゼロ付近が停滞・増加領域。",
            "top15_segments": top.to_dict(orient="records"),
        },
        ensure_ascii=False,
        indent=2,
    )


@function_tool
def analyze_sapoca_effect() -> str:
    """
    「人対車両×夜間」シナリオに絞り、サポカー有無で歩行者（当事者B）死亡率を比較する。
    注意: サポカー列は2024年データのみ存在。
    """
    df = load_honhyo(2024)

    subset = df[
        (df["事故類型"] == 1)
        & (df["夜間フラグ"])
        & (df["サポカー名A"].isin(["サポカー", "非サポカー"]))
    ].copy()

    subset["B死亡"] = (df.loc[subset.index, "人身損傷程度（当事者B）"] == 1).astype(int)

    result = (
        subset.groupby("サポカー名A")
        .agg(事故件数=("B死亡", "count"), B死亡件数=("B死亡", "sum"))
        .assign(B死亡率=lambda x: (x["B死亡件数"] / x["事故件数"] * 100).round(2))
    )

    return json.dumps(
        {
            "対象": "人対車両×夜間（2024年）",
            "サポカー別B死亡率": result.to_dict(),
            "注意": (
                "サポカーの方が死亡率が高く見えても、それは残存事故バイアスの可能性がある。"
                "AEBが軽微な事故を防いだ結果、統計に残るのは「センサー限界を超えた重篤事故」のみ。"
                "因果効果の評価にはITARDA No.133のような保有台数ベース比較が必要。"
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


@function_tool
def project_2030() -> str:
    """
    2020→2024の実績トレンドを2030年まで延長し、目標（1,392件）との乖離を試算する。
    「自然更新だけでは届かない」という構造的課題を数値で示す。
    """
    d20 = load_honhyo(2020)
    d24 = load_honhyo(2024)

    fatal_2020 = int((d20["死者数"] > 0).sum())
    fatal_2024 = int((d24["死者数"] > 0).sum())
    annual_trend = (fatal_2024 - fatal_2020) / 4
    projected_2030 = fatal_2024 + annual_trend * 6
    target_2030 = 1392
    gap = projected_2030 - target_2030

    return json.dumps(
        {
            "実績": {"2020年死亡事故件数": fatal_2020, "2024年死亡事故件数": fatal_2024},
            "年間トレンド(件/年)": round(annual_trend, 1),
            "2030年推計値": round(projected_2030),
            "2030年目標値": target_2030,
            "ギャップ（推計-目標）": round(gap),
            "結論": (
                f"現行ペースでは2030年に約{round(projected_2030)}件。"
                f"目標{target_2030}件まで約{round(gap)}件のギャップが残る。"
                "自然更新だけでは目標に届かず、能動的な施策介入が必要。"
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


@function_tool
def analyze_nonsapoca_rate() -> str:
    """
    死亡事故（死者数>0）における非サポカー車両の比率を計算する。
    四輪当事者A（乗用車・貨物車等）に絞って集計する。
    注意: サポカー列は2024年データのみ存在。
    """
    df = load_honhyo(2024)
    fatal = df[df["死者数"] > 0].copy()

    # 当事者種別 11-29 が四輪系（乗用車・貨物車・特殊車）
    fatal_4w = fatal[fatal["当事者種別（当事者A）"].between(11, 29)]
    sapoca_dist = fatal_4w["サポカー名A"].value_counts()
    total_4w = len(fatal_4w)
    nonsapoca = int(sapoca_dist.get("非サポカー", 0))
    rate = nonsapoca / total_4w * 100 if total_4w > 0 else 0

    return json.dumps(
        {
            "対象": "2024年死亡事故の四輪当事者A",
            "総件数": total_4w,
            "サポカー分布": sapoca_dist.to_dict(),
            "非サポカー比率(%)": round(rate, 1),
            "解釈": (
                f"死亡事故加害車両（四輪）の{round(rate, 1)}%は非サポカー。"
                "フリートの自然更新には10〜15年単位の時間がかかる。"
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


@function_tool
def analyze_environmental_factors() -> str:
    """
    天候（雨・雪等）や地形（市街地・非市街地）が致死率や死亡事故件数に与える影響を分析する。
    「非市街地の方が致死率が高い」等の環境要因を確認するために使う。
    """
    df = load_both_years()
    df["死亡フラグ"] = (df["死者数"] > 0).astype(int)

    env_stats = (
        df.groupby(["地形名", "天候名"])
        .agg(
            全事故件数=("死亡フラグ", "count"),
            死亡事故件数=("死亡フラグ", "sum"),
        )
        .assign(致死率=lambda x: (x["死亡事故件数"] / x["全事故件数"] * 100).round(3))
        .reset_index()
    )

    return json.dumps(
        {
            "説明": "地形・天候別の事故統計。非市街地や悪天候時のリスクを定量化する。",
            "environment_stats": env_stats.to_dict(orient="records"),
        },
        ensure_ascii=False,
        indent=2,
    )


@function_tool
def filter_accidents(year: int, weather: str = None, day_night: str = None) -> str:
    """
    指定された条件（年、天候、昼夜）で事故データをフィルタリングし、集計結果を返す。
    year: 2020 または 2024
    weather: '晴', '曇', '雨', '霧', '雪' のいずれか（任意）
    day_night: '昼間' または '夜間'（任意）
    """
    df = load_honhyo(year)
    
    if weather:
        df = df[df["天候名"] == weather]
    
    if day_night:
        is_night = (day_night == "夜間")
        df = df[df["夜間フラグ"] == is_night]

    total = len(df)
    fatal_accidents = int((df["死者数"] > 0).sum())
    total_deaths = int(df["死者数"].sum())
    
    return json.dumps(
        {
            "year": year,
            "condition": {"weather": weather, "day_night": day_night},
            "全事故件数": total,
            "死亡事故件数": fatal_accidents,
            "死者数合計": total_deaths,
            "致死率(%)": round(fatal_accidents / total * 100, 3) if total > 0 else 0
        },
        ensure_ascii=False,
        indent=2
    )


@function_tool
def aggregate_accidents(year: int, groupby: list[str], weather: str = None, day_night: str = None, road_shape: str = None) -> str:
    """
    指定された年において、特定のカラムでグループ化して集計を行う（SQLのGROUP BYに相当）。
    year: 2020 または 2024
    groupby: グループ化するカラム名のリスト。利用可能なカラム: ['事故類型名', '昼夜名', '天候名', '地形名', '道路形状名', '速度帯A']
    weather: '晴', '曇', '雨', '霧', '雪' のいずれか（任意フィルタ）
    day_night: '昼間' または '夜間'（任意フィルタ）
    road_shape: '交差点', 'トンネル', '橋', 'カーブ', '単路-その他' のいずれか（任意フィルタ）
    """
    df = load_honhyo(year)
    
    # フィルタリング適用
    if weather:
        df = df[df["天候名"] == weather]
    if day_night:
        df = df[df["夜間フラグ"] == (day_night == "夜間")]
    if road_shape:
        df = df[df["道路形状名"] == road_shape]

    if not groupby:
        return "Error: groupby column list is empty."

    # 有効なカラムかチェック
    valid_cols = [c for c in groupby if c in df.columns]
    if not valid_cols:
        return f"Error: No valid columns found in {groupby}. Available: {list(df.columns)}"

    agg = (
        df.groupby(valid_cols)
        .agg(
            全事故件数=("死者数", "count"),
            死亡事故件数=("死者数", lambda x: (x > 0).sum()),
            死者数合計=("死者数", "sum")
        )
        .assign(致死率=lambda x: (x["死亡事故件数"] / x["全事故件数"] * 100).round(3))
        .reset_index()
    )

    return json.dumps(
        {
            "year": year,
            "groupby": valid_cols,
            "filters": {"weather": weather, "day_night": day_night},
            "results": agg.to_dict(orient="records")
        },
        ensure_ascii=False,
        indent=2
    )


ALL_TOOLS = [
    get_overview,
    compare_years,
    analyze_fatal_scenarios,
    analyze_sapoca_effect,
    project_2030,
    analyze_nonsapoca_rate,
    analyze_environmental_factors,
    filter_accidents,
    aggregate_accidents,
]
