# -*- coding: utf-8 -*-
"""
============================================================
城市租房数据集分析 · 租金影响因素建模
============================================================

【这个脚本解决什么】
    回答 Q2：哪些属性显著影响租金，影响幅度多大。
    方法是对 log(租金) 做多元线性回归，系数可以直接翻译成
    百分比溢价（exp(β) - 1），比树模型的特征重要性更好解释。

【为什么不用随机森林】
    当前环境只有 pandas + numpy，没有 sklearn。
    与其堆一个跑不起来的树模型，不如把一个能跑、系数可解释的
    回归做扎实。面试被问到时，这是站得住的取舍。

【运行】
    python modelling.py
============================================================
"""

import numpy as np
import pandas as pd

CLEAN_CSV = "data/rent_clean.csv"
OUT_CSV = "data/model_results.csv"

# 类别变量的基准组（每组第一个）。回归系数是相对基准组的差异。
BASE_DISTRICT_MIN = 100     # 区县样本量少于这个数就并入「其他」，避免系数不稳


def load_data():
    """读清洗后的数据，并排除合租样本。"""
    df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    n0 = len(df)

    # 合租的"面积"是单间面积、"租金"是单间价格，与整租口径不同，
    # 混在一起建模会让"面积"的系数失真。样本量小，直接排除。
    if "is_shared" in df.columns:
        df = df[~df["is_shared"].astype(bool)].copy()
        print(f"[i] 排除合租样本 {n0 - len(df)} 条，剩余 {len(df)} 条")

    df = df[df["rent"] > 0].copy()
    df["log_rent"] = np.log(df["rent"])
    df["log_area"] = np.log(df["area"])
    return df


def build_features(df):
    """
    构造特征矩阵。

    设计选择：
      1) 用 log(area) 而不是 area —— 面积与租金不是线性关系，
         取对数后系数即为"面积弹性"（面积涨 1%，租金涨 β%）。
      2) 类别变量做 one-hot，并各丢一个基准组（加截距时不丢会共线）。
      3) distance 缺失 43%，填中位数之外再加一个缺失标记，
         让模型区分"离地铁 X 米"和"数据里没写"。
    """
    num_cols = ["log_area", "room", "hall", "bath"]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0

    df["has_distance"] = df["distance"].notna().astype(int)
    df["distance_filled"] = df["distance"].fillna(df["distance"].median())

    # 区县太碎就归并，否则系数是噪音
    if df["district"].nunique() > 12:
        cnt = df["district"].value_counts()
        keep = cnt[cnt >= BASE_DISTRICT_MIN].index
        df["district_g"] = df["district"].where(df["district"].isin(keep), "其他")
    else:
        df["district_g"] = df["district"]

    # 关键：city 与 district 不能同时进模型。
    # 区县本身从属于城市，两者高度共线，同时放会让系数互相"抢"解释力，
    # 实测会出现"北京比上海低 25%"这种与描述统计直接矛盾的系数。
    # 这里保留 district（粒度更细、解释力更强），城市层面的差异用分组统计讲。
    cat_cols = [c for c in ["decoration", "south_facing", "district_g"]
                if c in df.columns]

    y = df["log_rent"].values
    X_parts = [df[["log_area", "room", "hall", "bath",
                   "distance_filled", "has_distance"]].astype(float).values]
    names = ["log_area", "room", "hall", "bath", "distance", "has_distance"]

    bases = {}
    for c in cat_cols:
        s = df[c].astype(str).fillna("缺失")
        vals = sorted(s.unique())
        bases[c] = vals[0] if vals else ""
        dummies = pd.get_dummies(s, prefix=c, drop_first=True, dtype=float)
        X_parts.append(dummies.values)
        names.extend(dummies.columns.tolist())

    X = np.column_stack(X_parts)
    # 去掉方差为 0 的列（某一类只出现一次时 dummy 会退化成常量）
    keep_mask = X.std(axis=0) > 0
    X = X[:, keep_mask]
    names = [n for n, k in zip(names, keep_mask) if k]

    return X, y, names, bases


def ols(X, y, names):
    """
    最小二乘回归。加截距后用 lstsq 求解。
    返回系数、拟合值、残差。
    """
    n = len(y)
    X1 = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    return beta, pred, y - pred


def evaluate(y_true_log, pred_log, rent_true):
    """在 log 尺度算 R²，在原始尺度算 MAE 与 MAPE。"""
    ss_res = ((y_true_log - pred_log) ** 2).sum()
    ss_tot = ((y_true_log - y_true_log.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    pred_rent = np.exp(pred_log)
    mae = np.abs(pred_rent - rent_true).mean()
    mape = (np.abs(pred_rent - rent_true) / rent_true).mean() * 100
    return r2, mae, mape


def main():
    df = load_data()
    print(f"[i] 建模样本 {len(df)} 条\n")

    # 固定随机种子，保证结果可复现
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.8)
    tr, te = idx[:cut], idx[cut:]

    X, y, names, bases = build_features(df)
    print(f"[i] 特征维度 {X.shape[1]}")
    print(f"[i] 各类别变量的基准组：{bases}\n")

    beta, pred, resid = ols(X[tr], y[tr], names)

    # 训练集与测试集分别评估
    r2_tr = 1 - ((y[tr] - pred) ** 2).sum() / ((y[tr] - y[tr].mean()) ** 2).sum()
    X1_te = np.column_stack([np.ones(len(te)), X[te]])
    pred_te = X1_te @ beta
    r2_te, mae_te, mape_te = evaluate(y[te], pred_te, df["rent"].values[te])

    print("=" * 66)
    print("模型评估（目标变量 = log(租金)）")
    print("=" * 66)
    print(f"  训练集 R²      {r2_tr:.4f}")
    print(f"  测试集 R²      {r2_te:.4f}")
    print(f"  测试集 MAE     {mae_te:,.0f} 元/月")
    print(f"  测试集 MAPE    {mape_te:.1f} %")
    print()
    print("  解读：R² 是在 log 尺度上的，MPE 反映的是原始金额的百分比误差。")
    print("  住房租金截面数据 R² 在 0.6-0.8 属正常，因为房龄、采光、房东心态")
    print("  等大量因素不在数据里。")

    # ---- 系数解释 ----
    print("\n" + "=" * 66)
    print("回归系数（相对基准组的变化）")
    print("=" * 66)
    coef = pd.DataFrame({"特征": ["截距"] + names, "系数": beta})
    coef["百分比效应"] = (np.exp(coef["系数"]) - 1) * 100
    coef = coef[coef["特征"] != "截距"].copy()

    # 面积弹性单独说明
    ela = coef[coef["特征"] == "log_area"]["系数"].iloc[0]
    print(f"\n  【面积弹性】{ela:.3f} —— 面积每增加 1%，租金平均增加 {ela:.2f}%")
    print("  （弹性小于 1 说明存在规模效应：面积越大，单位面积越便宜）")

    print(f"\n  【区县效应】基准 = 样本量最大的基准区县（下表为绝对值最大的几项）")
    dist_rows = coef[coef["特征"].str.startswith("district_g_")].copy()
    if len(dist_rows):
        dist_rows["_abs"] = dist_rows["百分比效应"].abs()
        for _, r in dist_rows.sort_values("_abs", ascending=False).head(10).iterrows():
            print(f"    {r['特征'][11:]:<8} {r['百分比效应']:>+7.1f}%")

    print(f"\n  【装修效应】基准 = 其他装修状态")
    for _, r in coef[coef["特征"].str.startswith("decoration_")].iterrows():
        print(f"    {r['特征'][11:]:<6} {r['百分比效应']:>+7.1f}%")

    print(f"\n  【朝向效应】")
    for _, r in coef[coef["特征"].str.startswith("south_facing_")].iterrows():
        print(f"    {r['特征'][13:]:<8} {r['百分比效应']:>+7.1f}%")

    print(f"\n  【户型效应】每多一间房 / 一个卫生间，租金变化：")
    for k in ["room", "hall", "bath"]:
        row = coef[coef["特征"] == k]
        if len(row):
            print(f"    {k:<6} {row['百分比效应'].iloc[0]:>+7.1f}%")

    print(f"\n  【地铁距离】每远离地铁 1000 米，租金变化：")
    d_row = coef[coef["特征"] == "distance"]
    if len(d_row):
        print(f"    {(np.exp(d_row['系数'].iloc[0] * 1000) - 1) * 100:>+7.1f}%")

    # ---- 特征贡献排序（按标准化系数）----
    print("\n" + "=" * 66)
    print("特征影响力排序（标准化系数绝对值）")
    print("=" * 66)
    sd = X.std(axis=0)
    std_beta = beta[1:] * sd
    order = np.argsort(-np.abs(std_beta))
    rank = pd.DataFrame({
        "特征": [names[i] for i in order],
        "标准化系数": [std_beta[i] for i in order],
    }).head(12)
    rank["影响方向"] = np.where(rank["标准化系数"] > 0, "正向", "负向")
    print(rank.round(4).to_string(index=False))

    coef.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[+] 系数明细已保存 {OUT_CSV}")

    # 残差诊断：哪些房源估不准
    print("\n" + "=" * 66)
    print("残差诊断：最容易被高估 / 低估的房源类型")
    print("=" * 66)
    te_df = df.iloc[te].copy()
    te_df["pred_rent"] = np.exp(pred_te)
    te_df["resid_pct"] = (te_df["rent"] - te_df["pred_rent"]) / te_df["pred_rent"] * 100
    worst = te_df.reindex(te_df["resid_pct"].abs().sort_values(ascending=False).index)
    print("  实际租金比预测低得最多的（模型高估）：")
    print(worst.head(3)[["city", "district", "area", "rent", "pred_rent", "resid_pct"]]
          .round(1).to_string(index=False))
    print("\n  实际租金比预测高得最多的（模型低估）：")
    print(worst.tail(3)[["city", "district", "area", "rent", "pred_rent", "resid_pct"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
