# -*- coding: utf-8 -*-
"""
============================================================
城市租房数据集分析 · 数据清洗 + EDA 脚本
============================================================

【怎么用 · 三步】

    第 1 步  先体检。在命令行运行：
                 python cleaning_eda.py --inspect
             它会打印：数据形状、每列类型、缺失率、样例值，
             以及「自动猜测的列名映射」。把猜测结果看一眼，不准的改掉。

    第 2 步  填 COL_MAP。打开下面的 [1] 配置区，把每个内部标准名
             对应的「你的 CSV 列名」填进去。填完再跑一次 --inspect 确认。

    第 3 步  正式跑：
                 python cleaning_eda.py
             清洗结果写到 data/rent_clean.csv，
             清洗日志写到 data/cleaning_log.csv，
             图表写到 fig/ 目录。

【为什么先做体检】
    不看数据就写清洗规则，等于蒙着眼睛改代码。真实租房数据的脏法
    千奇百怪（"3200元/月"、"89㎡"、"2室1厅"、"暂无"、空字符串、
    全角空格……），你必须先看到实际形态，才知道该用哪种解析方式。

【依赖】
    pip install pandas matplotlib

【设计原则 · 三个「不」】
    1. 不静默丢数据 —— 每删一行都记进清洗日志，最后能对账。
    2. 不整段报错   —— 某列不存在时只警告、跳过，不让整个流程崩掉。
    3. 不覆盖原始数据 —— raw 只读，clean 由代码重新生成，过程可复现。
============================================================
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# 图表中文显示：Windows 下不设字体，图上中文会变成方块。
# matplotlib 设为可选依赖 —— 没装绘图库时清洗照常跑完，只是不出图。
try:
    import matplotlib
    matplotlib.use("Agg")            # 不弹窗，直接存文件，适合脚本批量跑
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[提示] 未安装 matplotlib，将跳过图表输出（不影响清洗与分析）")


# ============================================================
# [1] 配置区
# ============================================================

RAW_CSV = "data/rent_raw.csv"      # 原始数据，只读
CLEAN_CSV = "data/rent_clean.csv"  # 清洗后输出
LOG_CSV = "data/cleaning_log.csv"  # 清洗日志
FIG_DIR = "fig"                    # 图表输出目录

# ---- 列名映射 ----
# 格式：  "内部标准名": "你的CSV里的列名"
# 已按「北上广深链家租房数据集」的真实列名填好。换数据源时只改这张表。
COL_MAP = {
    "house_id":      "_id",                # 房源唯一编号
    "city":          "city",               # 城市（北上广深）
    "district":      "dist",               # 区县（朝阳 / 浦东 / 白云…）
    "business_area": "bizcircle_name",     # 商圈（做分层基准的关键列）
    "community":     "resblock_name",      # 小区名
    "rent":          "rent_price_listing", # 月租金（原始数据里是字符串）
    "area":          "rent_area",          # 面积（同上，是字符串）
    "layout_raw":    "layout",             # 户型，如 "3室2厅2卫"
    "orientation":   "frame_orientation",  # 朝向
    "floor_raw":     "",                   # 本数据集无楼层字段
    "decoration":    "",                   # 无独立装修列，从 house_tag 派生
    "elevator":      "",                   # 本数据集无电梯字段
}

# 关键词提示表：供自动猜测列名使用，改 COL_MAP 时不需要动这里
COLUMN_HINTS = {
    "house_id":      ["房源编号", "房源id", "house_id", "id", "编号"],
    "city":          ["城市", "city"],
    "district":      ["区域", "区县", "城区", "district", "area_name"],
    "business_area": ["商圈", "板块", "片区", "business", "block"],
    "community":     ["小区", "楼盘", "community", "village"],
    "rent":          ["租金", "月租", "价格", "售价", "rent", "price"],
    "area":          ["面积", "建筑面积", "area", "size"],
    "layout_raw":    ["户型", "房型", "layout", "room"],
    "orientation":   ["朝向", "orientation", "direction"],
    "floor_raw":     ["楼层", "所在楼层", "floor"],
    "decoration":    ["装修", "装修情况", "decoration", "fitment"],
    "elevator":      ["电梯", "elevator", "lift"],
}

# 缺失率阈值：低于 LOW 填充，高于 HIGH 直接丢弃整列
LOW_MISS = 0.05
HIGH_MISS = 0.30

# 丢列豁免：这些列的缺失往往代表"没有这个属性"，不是"数据丢了"，不能按缺失率丢
# 例：地铁距离为空，多半是该小区附近没有地铁站 —— 这本身就是信息
KEEP_ON_HIGH_MISS = ["distance"]

# 异常值分位阈值。用 1% / 99%，不用 3σ ——
# 租金是右偏分布，标准差法会把大量真实高租金误判为异常。
Q_LOW, Q_HIGH = 0.01, 0.99

# 分组样本量下限：低于这个数的分组，分位数不稳定，退回全城口径
MIN_GROUP_SIZE = 10

# 用于判断「同一房源重复挂牌」的业务键
DUP_KEY = ["community", "area", "layout_raw", "rent"]

_LOG = []   # 清洗日志缓冲区


def log(step, before, after, note=""):
    """记录每一步清洗前后的行数，最后统一落盘。这是作品集里最能体现严谨性的细节。"""
    _LOG.append({
        "step": step,
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "note": note,
    })
    print(f"  [日志] {step:<24} {before:>7} → {after:>7}  删 {before - after}")


# ============================================================
# [2] 数据体检 —— 动任何清洗之前先跑这个
# ============================================================

def suggest_mapping(df):
    """按关键词猜测列名映射，仅供人工确认，不要全信。"""
    guess = {}
    used = set()
    for std, hints in COLUMN_HINTS.items():
        for col in df.columns:
            if col in used:
                continue
            low = str(col).lower()
            if any(h.lower() in low for h in hints):
                guess[std] = col
                used.add(col)
                break
    return guess


def inspect_data(df):
    """打印数据全貌。判断清洗策略全靠这一屏输出。"""
    print("=" * 66)
    print(f"数据形状：{df.shape[0]} 行 × {df.shape[1]} 列")
    print("=" * 66)

    print("\n【逐列概览】")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "缺失数": df.isna().sum(),
        "缺失率": (df.isna().mean() * 100).round(2),
        "唯一值数": df.nunique(),
        "样例值": [str(df[c].dropna().iloc[0])[:28] if df[c].notna().any() else "（全空）"
                   for c in df.columns],
    })
    print(info.to_string())

    print("\n【完全重复行】", df.duplicated().sum(), "行")

    print("\n【列名自动猜测】← 对照你的实际列名，把错的改到 COL_MAP 里")
    for std, col in suggest_mapping(df).items():
        flag = "OK" if col == COL_MAP.get(std) else "已猜"
        print(f"  {std:<14} ← {col!r:<24} [{flag}]")
    missing = [s for s in COL_MAP if s not in suggest_mapping(df)]
    if missing:
        print(f"  未匹配到：{missing}  ← 这几个需要你手动指定（或该列不存在）")

    print("\n【数值列分布】")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print(df[num_cols].describe().T.to_string())
    else:
        print("  无原生数值列 —— 说明租金/面积都是字符串，需先做数值解析")
    print()


# ============================================================
# [3] 列名标准化
# ============================================================

def standardize_columns(df):
    """
    把 CSV 的列名统一成内部标准名。

    为什么要做这一步：后面的清洗逻辑全部依赖标准名。
    如果直接写 df["租金"]，脚本就和某一份具体数据绑死了，
    换一份数据就得全文替换。映射层让你只改一处。
    """
    mapping = {v: k for k, v in COL_MAP.items() if v in df.columns}
    df = df.rename(columns=mapping)
    extra = [c for c in df.columns if c not in COL_MAP]
    if extra:
        print(f"  [提示] 以下列未纳入标准名，将保留但不参与分析：{extra}")
    return df


# ============================================================
# [4] 类型转换 —— 必须在去重和异常值之前做
# ============================================================

def parse_number(series):
    """
    从混合文本里抽数字。

    "3200元/月" → 3200.0
    "89.5㎡"    → 89.5
    "1,200"     → 1200.0
    "暂无"      → NaN
    "--"        → NaN

    关键点：errors="coerce" 保证解析失败时变 NaN 而不是抛异常，
    否则一条 "面议" 就能让整列转换崩掉。
    """
    if series.dtype.kind in "if":          # 已经是数值，直接返回
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)   # 千分位
        .str.replace("，", "", regex=False)  # 中文逗号
        .str.replace(" ", "", regex=False)   # 空格
    )
    return pd.to_numeric(cleaned.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")


def parse_layout(series):
    """
    从 "2室1厅1卫" 拆出 室/厅/卫 三个数。

    为什么不在采集时拆：原始串要保留。清洗是可重跑的，
    原始信息一旦丢了，后面想改解析规则就没数据可改。
    """
    s = series.astype(str)
    out = pd.DataFrame(index=series.index)
    out["room"] = pd.to_numeric(s.str.extract(r"(\d+)\s*室")[0], errors="coerce")
    out["hall"] = pd.to_numeric(s.str.extract(r"(\d+)\s*厅")[0], errors="coerce")
    out["bath"] = pd.to_numeric(s.str.extract(r"(\d+)\s*卫")[0], errors="coerce")
    return out


def parse_floor(series):
    """
    楼层列常见两段信息："中楼层/共18层"、"低楼层(共6层)"。

    返回 (所在楼层, 总楼层)。若原值是"中/低/高"这类相对描述，
    所在楼层留空 —— 不要强行编一个数字，那是在造数据。
    """
    s = series.astype(str)
    floor = pd.to_numeric(s.str.extract(r"(\d+)\s*层")[0], errors="coerce")
    total = pd.to_numeric(s.str.extract(r"共\s*(\d+)\s*层")[0], errors="coerce")
    # 两段都抽到数字时，第一个才是"所在楼层"
    both = s.str.contains("共", na=False)
    floor = floor.where(~both, pd.to_numeric(s.str.extract(r"(\d+)\s*层.*?共")[0], errors="coerce"))
    return floor, total


def convert_types(df):
    """统一做类型转换。顺序：数值 → 派生字段 → 类别字段。"""
    before = len(df)

    for col in ["rent", "area"]:
        if col in df.columns:
            df[col] = parse_number(df[col])

    if "layout_raw" in df.columns:
        df = pd.concat([df, parse_layout(df["layout_raw"])], axis=1)

    if "floor_raw" in df.columns:
        df["floor"], df["total_floor"] = parse_floor(df["floor_raw"])
        # 楼层占比：比绝对楼层更有解释力（18层里的5层 ≠ 5层里的5层）
        df["floor_ratio"] = (df["floor"] / df["total_floor"]).where(
            df["total_floor"] > 0
        )

    # 类别字段转 category：省内存，且 groupby 时不会把拼写近似的值当数值
    for col in ["city", "district", "business_area", "community",
                "orientation", "decoration"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    for col in ["rent", "area"]:
        if col in df.columns:
            bad = df[col].isna().sum()
            if bad:
                print(f"  [提示] {col} 有 {bad} 条无法解析为数值（如'面议'/'暂无'）")

    log("类型转换", before, len(df))
    return df


# ============================================================
# [5] 去重 —— 要分两层，别只做 drop_duplicates
# ============================================================

def drop_duplicates(df):
    """
    第一层：完全重复行（所有字段一模一样）—— 多半是采集翻页重叠。
    第二层：同一房源重复挂牌 —— 不同中介挂同一套房，ID 不同但实体相同。

    第二层用业务键判断。注意这层是「疑似」，宁可人工看几眼再决定，
    因为同一个小区同面积同户型同租金，也可能是真的两套房子。
    """
    before = len(df)

    df = df.drop_duplicates()
    log("去重·完全重复行", before, len(df))

    mid = len(df)
    keys = [k for k in DUP_KEY if k in df.columns]
    if len(keys) == len(DUP_KEY):
        # 保留缺失最少的记录：一条完整记录比一条半残的更有分析价值
        df = df.assign(_miss=df[keys].isna().sum(axis=1))
        df = (df.sort_values("_miss")
                .drop_duplicates(subset=keys, keep="first")
                .drop(columns="_miss"))
        log("去重·房源业务键重复", mid, len(df), "键=" + "+".join(keys))
    else:
        print(f"  [跳过] 业务键不全（缺 {set(DUP_KEY) - set(keys)}），仅做完全重复去重")

    return df.reset_index(drop=True)


# ============================================================
# [6] 缺失值 —— 分级处理，不要一刀切
# ============================================================

def handle_missing(df):
    """
    三级策略：
      缺失率 > 30%  → 整列丢弃（信息量太低，填补等于造假）
      核心字段缺失  → 删行（rent/area 缺失就无法分析，补不了）
      其余低缺失    → 数值用中位数，类别用"未知"

    为什么数值用中位数不用均值：租金右偏，均值被高价房拉高，
    用均值填补会系统性高估普通房源。
    """
    before = len(df)

    miss = df.isna().mean()
    drop_cols = [c for c in miss[miss > HIGH_MISS].index
                 if c not in KEEP_ON_HIGH_MISS]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  [丢弃列] 缺失率超过 {HIGH_MISS:.0%}：{drop_cols}")
        # 派生列全空通常不是数据缺失，而是原始字段压根没有具体数值。
        # 例如楼层写成"中楼层(共18层)"，只能提出总层数，提不出层号。
        # 这种情况属于正常，别误以为自己代码写错了。
        derived = {"floor", "floor_ratio", "room", "hall", "bath"} & set(drop_cols)
        if derived:
            print(f"    ↑ {sorted(derived)} 是解析派生列。全空一般说明原始字段只有"
                  f"'中/低/高楼层'这类相对描述，提不出具体数值——这是正常现象，"
                  f"不影响其余字段的分析。")

    core = [c for c in ["rent", "area"] if c in df.columns]
    df = df.dropna(subset=core)
    log("缺失值·删核心字段缺失行", before, len(df), f"核心列={core}")

    mid = len(df)
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if df[col].dtype.kind in "if":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("未知")
    log("缺失值·填充其余字段", mid, len(df), "数值取中位数，类别填'未知'")

    return df


# ============================================================
# [7] 异常值 —— 分组判断，别用全城口径
# ============================================================

def handle_outliers(df):
    """
    三条规则，按顺序执行：
      1) 业务硬规则：面积 < 5㎡ 或 租金 <= 0 → 明显是录入错误或钓鱼盘
      2) 单价分组分位：以商圈为单位算 1%/99%，而不是全城
      3) 小分组退回全城口径：样本太少的商圈分位数不可信

    为什么强调分组：全城 1% 分位会把核心地段的大户型高价房判成异常，
    而按商圈判断，它是合理的。这是「异常值」和「极端但真实」的区别。

    注意这里用 IQR/分位而非 3σ，理由同缺失值：右偏分布下 σ 会被长尾撑大。
    """
    before = len(df)

    if "rent" in df.columns and "area" in df.columns:
        bad = (df["area"] < 5) | (df["rent"] <= 0)
        # 必须 .copy()：布尔索引取出的子集是视图，直接在上面赋值会触发
        # SettingWithCopyWarning，而且行为依赖 pandas 版本，不可靠。
        df = df[~bad].copy()
        log("异常值·业务硬规则", before, len(df), "面积<5㎡ 或 租金<=0")

    mid = len(df)
    if "rent" in df.columns and "area" in df.columns:
        df["unit_price"] = df["rent"] / df["area"]

        group_col = "business_area" if "business_area" in df.columns else None
        if group_col:
            size = df.groupby(group_col, observed=True)["unit_price"].transform("size")
            # 小分组用全城分位，大分组用自己分位
            lo_g = df.groupby(group_col, observed=True)["unit_price"].transform(
                lambda s: s.quantile(Q_LOW))
            hi_g = df.groupby(group_col, observed=True)["unit_price"].transform(
                lambda s: s.quantile(Q_HIGH))
            lo_a = df["unit_price"].quantile(Q_LOW)
            hi_a = df["unit_price"].quantile(Q_HIGH)
            lo = lo_g.where(size >= MIN_GROUP_SIZE, lo_a)
            hi = hi_g.where(size >= MIN_GROUP_SIZE, hi_a)
            note = f"按 {group_col} 分组 {Q_LOW:.0%}/{Q_HIGH:.0%} 分位"
        else:
            lo = df["unit_price"].quantile(Q_LOW)
            hi = df["unit_price"].quantile(Q_HIGH)
            note = f"无商圈列，按全城 {Q_LOW:.0%}/{Q_HIGH:.0%} 分位"

        # between 保留区间内；boundary 外的拿出来单独存，便于复核
        keep = df["unit_price"].between(lo, hi)
        removed = df[~keep].copy()
        df = df[keep].copy()
        log("异常值·单价分位过滤", mid, len(df), note)

        if len(removed):
            os.makedirs("data", exist_ok=True)
            removed.head(200).to_csv("data/outliers_removed.csv",
                                     index=False, encoding="utf-8-sig")
            print(f"  [留档] 被剔除的异常样本前 200 条已存 data/outliers_removed.csv，"
                  f"建议人工看几眼")

    return df.reset_index(drop=True)


def derive_from_tags(df):
    """
    从 house_tag 派生字段。

    这份数据的装修信息没有独立列，只藏在标签串里（如"精装 集中供暖 双卫生间"）。

    关键判断：只有约两成房源带"精装"标签，其余**不是"非精装"，而是"未标注"**。
    房东没填不等于毛坯，所以单独作为一类，不要和"简装/毛坯"合并 ——
    否则会把"没写"当成"没装修"，直接污染溢价结论。
    """
    tag_col = "house_tag"
    if tag_col not in df.columns:
        return df

    tag = df[tag_col].fillna("").astype(str)
    dec = pd.Series("未标注", index=df.index, dtype="object")
    for kw in ["豪装", "精装", "简装", "毛坯"]:
        dec = dec.mask(tag.str.contains(kw), kw)
    df["decoration"] = dec

    # 地铁距离：原始是字符串，转数值后可用于分析"距地铁远近对租金的影响"
    if "distance" in df.columns:
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    return df


# ============================================================
# [8] 派生特征 —— 为 EDA 与分析问题准备
# ============================================================

def add_features(df):
    """
    派生特征全部来自已有字段，不引入外部知识，保证可复现。
    这些字段是后面 Q1-Q5 分析的基础，不建议跳过。
    """
    if {"rent", "area"}.issubset(df.columns) and "unit_price" not in df.columns:
        df["unit_price"] = df["rent"] / df["area"]

    if "area" in df.columns:
        # 面积分箱：用固定业务区间而不是等频分箱，
        # 因为"60㎡""90㎡"是租房市场的自然分界，评审看得懂
        df["area_bin"] = pd.cut(
            df["area"], bins=[0, 30, 60, 90, 120, np.inf],
            labels=["30以下", "30-60", "60-90", "90-120", "120以上"],
        )

    # 朝向归并：原始值是多朝向组合（"南 北"="南北通透"，各房间朝向不同），
    # 96 个取值直接分组毫无意义。归并成"是否朝南"才有分析价值。
    #
    # 注意正则：必须匹配**独立的**"南"。用 contains("南") 会把"东南""西南"
    # 也算成朝南，这是很容易犯的错 —— 那两个字是东南向，不是南向。
    if "orientation" in df.columns:
        o = df["orientation"].astype(str)
        is_south = o.str.contains(r"(?:^|\s)南(?:\s|$)", regex=True)
        sf = pd.Series(pd.NA, index=df.index, dtype="object")
        sf[is_south] = True
        sf[(~is_south) & (o != "未知")] = False
        df["south_facing"] = sf

    if "elevator" in df.columns:
        # 转成三态布尔：True=有，False=无，NA=未知。
        # 注意 NA 不等于 False —— "不清楚有没有电梯"和"确认没电梯"是两回事，
        # 分析时不要图省事用 fillna(False) 合并，那会污染"无电梯"组的结论。
        df["elevator"] = df["elevator"].astype("string").str.extract(
            r"(有|无|是|否|1|0)")[0].map(
            {"有": True, "是": True, "1": True, "无": False, "否": False, "0": False}
        ).astype("boolean")

    df["collect_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")

    # 整租 / 合租标记。
    # 重要：合租房源填的"面积"是单间面积，"租金"是单间价格，与整租口径不同，
    # 两者的单价不能直接比较。合租单价看起来更高，是口径问题，不是市场规律。
    # 做单价分析时要么排除合租，要么分开看。
    if "type" in df.columns:
        df["is_shared"] = df["type"].astype(str).str.contains("合租").fillna(False)

    return df


# ============================================================
# [9] EDA —— 描述统计与分组统计
# ============================================================

def eda_describe(df):
    """单变量描述：看分布形状，判断该用什么统计量。"""
    print("\n" + "=" * 66)
    print("EDA 1｜单变量描述统计")
    print("=" * 66)

    num_cols = [c for c in ["rent", "area", "unit_price", "floor_ratio"]
                if c in df.columns]
    if not num_cols:
        print("无可用数值列")
        return

    desc = df[num_cols].describe().T
    desc["偏度"] = df[num_cols].skew()
    desc["峰度"] = df[num_cols].kurt()
    desc["变异系数"] = (desc["std"] / desc["mean"]).round(3)
    print(desc.round(2).to_string())
    print("\n解读提示：偏度 > 1 说明右偏（少数高价拉长尾巴），"
          "此时应看中位数而非均值。")

    # 分布图
    os.makedirs(FIG_DIR, exist_ok=True)
    if "unit_price" in df.columns and HAS_PLOT:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        df["unit_price"].hist(bins=50, ax=axes[0], color="#4a9d8f")
        axes[0].set_title("租金单价分布（原始）")
        axes[0].set_xlabel("元/㎡/月")
        np.log1p(df["unit_price"]).hist(bins=50, ax=axes[1], color="#4a9d8f")
        axes[1].set_title("取对数后")
        axes[1].set_xlabel("log(1+单价)")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/01_unit_price_dist.png", dpi=130)
        plt.close()
        print(f"[图] {FIG_DIR}/01_unit_price_dist.png")


def eda_groupby(df):
    """分组统计：区位是租金第一解释变量，这是核心切面。"""
    print("\n" + "=" * 66)
    print("EDA 2｜分区位统计（Q1 租金空间格局）")
    print("=" * 66)

    # 多城市数据先看城市层 —— 跨城市差异通常比城市内区县差异更大
    if "city" in df.columns and df["city"].nunique() > 1 and "unit_price" in df.columns:
        print("\n【城市对比】← 跨城市是第一层切面")
        cty = (df.groupby("city", observed=True)
                 .agg(房源数=("unit_price", "size"),
                      中位单价=("unit_price", "median"),
                      中位租金=("rent", "median"),
                      中位面积=("area", "median"))
                 .sort_values("中位单价", ascending=False))
        print(cty.round(2).to_string())
        top, bot = cty["中位单价"].iloc[0], cty["中位单价"].iloc[-1]
        if bot:
            print(f"\n最高与最低城市的中位单价相差 {top / bot:.2f} 倍"
                  f"（{cty.index[0]} {top:.1f} vs {cty.index[-1]} {bot:.1f} 元/㎡/月）")

    group_col = "district" if "district" in df.columns else None
    if not group_col or "unit_price" not in df.columns:
        print("缺 district 或 unit_price，跳过")
        return

    stat = (df.groupby(group_col, observed=True)
              .agg(房源数=("unit_price", "size"),
                   中位单价=("unit_price", "median"),
                   均值单价=("unit_price", "mean"),
                   标准差=("unit_price", "std"),
                   中位租金=("rent", "median"),
                   中位面积=("area", "median"))
              .sort_values("中位单价", ascending=False))
    stat["离散度"] = (stat["标准差"] / stat["均值单价"]).round(3)
    print(stat.round(2).to_string())
    print("\n解读提示：中位单价排序 = 租金梯度；离散度高说明该区内部分化大，"
          "只看区均值会掩盖板块差异。")

    if "business_area" in df.columns:
        print("\n【商圈级价格带】← Q3 性价比分析的基准单位")
        ba = (df.groupby("business_area", observed=True)
                .agg(房源数=("unit_price", "size"),
                     中位单价=("unit_price", "median"),
                     P25=("unit_price", lambda s: s.quantile(.25)),
                     P75=("unit_price", lambda s: s.quantile(.75)))
                .sort_values("中位单价", ascending=False))
        # 注意：这里不用 .query()，因为 query 对中文列名支持不佳，容易报错
        ba = ba[ba["房源数"] >= MIN_GROUP_SIZE]
        print(ba.round(2).head(15).to_string())
        print(f"\n（仅显示样本量 ≥ {MIN_GROUP_SIZE} 的商圈，共 {len(ba)} 个；"
              f"小样本商圈的价格带不可信）")


def eda_categorical(df):
    """类别变量：看结构占比与组间单价差异。"""
    print("\n" + "=" * 66)
    print("EDA 3｜类别变量分组（Q2 属性溢价）")
    print("=" * 66)

    for col in ["decoration", "south_facing", "type", "area_bin"]:
        if col not in df.columns:
            continue
        # dropna=False 很重要：elevator 里的缺失是"未知"，不是"无电梯"。
        # 默认分组会把 NA 整组丢掉，导致各处占比加起来不到 100%，看着像 bug。
        g = (df.groupby(col, observed=True, dropna=False)
               .agg(房源数=("unit_price", "size"),
                    中位单价=("unit_price", "median"))
               .sort_values("中位单价", ascending=False))
        g["占比%"] = (g["房源数"] / len(df) * 100).round(1)
        print(f"\n【{col}】")
        print(g.round(2).to_string())

    print("\n解读提示：中位单价差异 ≠ 溢价幅度。装修好的房源往往面积更大、"
          "区位更好，直接比较会混淆因素。真正的溢价要用回归控制其他变量后再看（Q2 建模阶段）。")


def eda_relation(df):
    """双变量关系：检验线性和拐点。"""
    print("\n" + "=" * 66)
    print("EDA 4｜相关性（Q4 面积与单价的关系）")
    print("=" * 66)

    num_cols = [c for c in ["rent", "area", "unit_price", "floor_ratio", "room"]
                if c in df.columns]
    if len(num_cols) >= 2:
        # 注意：不要写 df.corr(numeric_only=True)，该参数在 pandas 1.5 才加入，
        # 旧版本会直接报 TypeError。先选数值列再 corr() 兼容所有版本。
        print(df[num_cols].corr().round(3).to_string())
        print("\n解读提示：若 area 与 unit_price 负相关，说明存在规模效应"
              "（大房子单位面积更便宜），这是 Q4 要检验的拐点前提。")

    if "area" in df.columns and "unit_price" in df.columns and HAS_PLOT:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.scatter(df["area"], df["unit_price"], s=8, alpha=.35, color="#4a9d8f")
        ax.set_xlabel("面积（㎡）")
        ax.set_ylabel("租金单价（元/㎡/月）")
        ax.set_title("面积与租金单价")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/02_area_vs_unitprice.png", dpi=130)
        plt.close()
        print(f"[图] {FIG_DIR}/02_area_vs_unitprice.png")


def eda_crosstab(df):
    """交叉表：两个类别变量的联合分布。"""
    if {"district", "area_bin"}.issubset(df.columns):
        print("\n" + "=" * 66)
        print("EDA 5｜交叉分布（区县 × 面积段 房源数）")
        print("=" * 66)
        ct = pd.crosstab(df["district"], df["area_bin"])
        print(ct.to_string())
        print("\n解读提示：某区在特定面积段房源特别集中，说明该区供给结构单一，"
              "可能是该区租金特征的成因之一。")


# ============================================================
# [10] 主流程
# ============================================================

def print_mapping_suggestion(raw_cols):
    """关键列缺失时，打印一份可直接复制的 COL_MAP 建议，省得用户手敲。"""
    guess, used = {}, set()
    for std, hints in COLUMN_HINTS.items():
        for col in raw_cols:
            if col in used:
                continue
            if any(h.lower() in str(col).lower() for h in hints):
                guess[std] = col
                used.add(col)
                break

    print("    可按下面这段改（左边是标准名不能动，右边换成你的真实列名）：")
    print()
    print("    COL_MAP = {")
    for std in COL_MAP:
        val = guess.get(std, "")
        pad = " " * max(1, 13 - len(std))
        print(f'        "{std}":{pad}"{val}",')
    print("    }")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", help="只体检，不做清洗")
    parser.add_argument("--in", dest="infile", default=RAW_CSV)
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        print(f"[x] 找不到文件：{args.infile}")
        print("    把 CSV 放到 data/rent_raw.csv，或用 --in 指定路径：")
        print("    python cleaning_eda.py --in 你的文件.csv --inspect")
        sys.exit(1)

    # 编码兼容：先试 utf-8，失败退回 gbk（中文 CSV 常见两种）
    try:
        df = pd.read_csv(args.infile, encoding="utf-8")
    except UnicodeDecodeError:
        print("[提示] utf-8 读取失败，改用 gbk 重试")
        df = pd.read_csv(args.infile, encoding="gbk")
    print(f"[+] 已读取 {args.infile}\n")

    if args.inspect:
        inspect_data(df)
        print("=" * 66)
        print("下一步：把上面「列名自动猜测」的结果对照实际列名，改好 [1] 配置区的")
        print("COL_MAP，再运行 python cleaning_eda.py 走完整流程。")
        return

    print("=" * 66)
    print("开始清洗")
    print("=" * 66)
    n0 = len(df)
    raw_cols = list(df.columns)          # 留一份原始列名，报错时要用

    df = standardize_columns(df)

    # ---- 关键列校验 ----
    # 这一步不能省。列名没对上时，后面的清洗会在"找不到列"的静默状态下一路跳过，
    # 最后打印"清洗完成"，而实际什么都没做 —— 这是最坏的失败方式：
    # 你以为拿到干净数据，其实把原始数据原样拿去分析了。
    missing_key = [c for c in ("rent", "area") if c not in df.columns]
    if missing_key:
        print()
        print("=" * 66)
        print(f"[x] 缺少关键列：{'、'.join(missing_key)} —— 已中止，避免产出假的清洗结果")
        print("=" * 66)
        print(f"你的 CSV 实际列名：{raw_cols}")
        print()
        print_mapping_suggestion(raw_cols)
        print("    填好之后重新运行本脚本。")
        sys.exit(1)

    df = convert_types(df)
    df = derive_from_tags(df)
    df = drop_duplicates(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    df = add_features(df)

    os.makedirs("data", exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(_LOG).to_csv(LOG_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 66)
    print(f"清洗完成：{n0} → {len(df)} 行（保留 {len(df)/n0:.1%}）")
    print(f"  清洗结果 {CLEAN_CSV}")
    print(f"  清洗日志 {LOG_CSV}   ← 每一步删了多少行都记在这里，建议写进报告")
    print("=" * 66)

    eda_describe(df)
    eda_groupby(df)
    eda_categorical(df)
    eda_relation(df)
    eda_crosstab(df)

    print("\n" + "=" * 66)
    print("EDA 完成。图表在 fig/ 目录。")
    print("下一步建议：把上面的分组结果对照 Q1/Q2 业务问题，写进分析报告。")
    print("=" * 66)


if __name__ == "__main__":
    main()
