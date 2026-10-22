# -*- coding: utf-8 -*-
"""
生成报告图表（纯 SVG，不依赖 matplotlib）
输出到 fig/ 目录。
"""
import os

import numpy as np
import pandas as pd

FIG_DIR = "fig"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv("data/rent_clean.csv", encoding="utf-8-sig")
if "is_shared" in df.columns:
    df_all = df.copy()
    df = df[~df["is_shared"].astype(bool)]

FONT = "'Microsoft YaHei','PingFang SC',sans-serif"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
ACC, ACC2 = "#0f766e", "#94a3b8"


def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>')


def text(x, y, s, size=13, color=INK, anchor="start", weight="normal", rotate=None):
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{tr}>{s}</text>')


def bar_chart(labels, values, title, path, unit="", w=820, h=420,
              color=ACC, highlight=None, fmt="{:.1f}"):
    """竖向柱状图。highlight = 需要强调的索引集合。"""
    ml, mr, mt, mb = 62, 30, 66, 78
    pw, ph = w - ml - mr, h - mt - mb
    vmax = max(values) * 1.18
    n = len(values)
    slot = pw / n
    bw = min(58, slot * 0.62)

    out = [svg_open(w, h)]
    out.append(text(ml, 34, title, 17, INK, weight="700"))
    if unit:
        out.append(text(ml, 54, unit, 12, MUTED))

    # 网格线与 y 轴刻度
    for i in range(5):
        v = vmax * i / 4
        y = mt + ph - ph * i / 4
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(text(ml - 8, y + 4, f"{v:.0f}", 11, MUTED, anchor="end"))

    for i, (lb, v) in enumerate(zip(labels, values)):
        cx = ml + slot * i + slot / 2
        bh = ph * v / vmax
        y = mt + ph - bh
        c = ACC if (highlight is None or i in highlight) else ACC2
        out.append(f'<rect x="{cx-bw/2:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                   f'height="{bh:.1f}" fill="{c}" rx="3"/>')
        out.append(text(cx, y - 7, fmt.format(v), 12, INK, anchor="middle", weight="700"))
        rot = -32 if n > 6 else 0
        out.append(text(cx, mt + ph + 20, lb, 12, MUTED, anchor="end" if rot else "middle",
                        rotate=rot))

    out.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(out))
    print("[图]", path)


def hbar_chart(labels, values, title, path, unit="", w=760, h=None):
    """横向条形图，用于系数效应（正负都可能）。"""
    n = len(labels)
    h = h or (86 + n * 34)
    ml, mr, mt = 96, 74, 62
    pw = w - ml - mr
    row = 30
    vmax = max(abs(min(values)), abs(max(values))) * 1.35

    out = [svg_open(w, h)]
    out.append(text(ml - 40, 32, title, 16, INK, weight="700"))
    if unit:
        out.append(text(ml - 40, 52, unit, 12, MUTED))

    zero = ml + pw / 2
    out.append(f'<line x1="{zero}" y1="{mt-8}" x2="{zero}" y2="{mt+n*row}" '
               f'stroke="{GRID}" stroke-width="1.5"/>')

    for i, (lb, v) in enumerate(zip(labels, values)):
        y = mt + i * row
        blen = (pw / 2) * abs(v) / vmax
        x = zero if v >= 0 else zero - blen
        c = ACC if v >= 0 else "#b45309"
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{blen:.1f}" height="17" '
                   f'fill="{c}" rx="2" opacity="0.88"/>')
        out.append(text(ml - 12, y + 13, lb, 12.5, INK, anchor="end"))
        sign = "+" if v > 0 else ""
        tx = zero + blen + 8 if v >= 0 else zero - blen - 8
        out.append(text(tx, y + 13, f"{sign}{v:.1f}%", 12, INK,
                        anchor="start" if v >= 0 else "end", weight="700"))

    out.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(out))
    print("[图]", path)


def hist_chart(series, title, path, bins=40, w=820, h=380, unit=""):
    """直方图。"""
    ml, mr, mt, mb = 62, 30, 66, 56
    pw, ph = w - ml - mr, h - mt - mb
    # 用分位裁剪长尾，避免极端值把所有柱子压扁
    lo, hi = series.quantile(0.005), series.quantile(0.995)
    s = series[(series >= lo) & (series <= hi)]
    cnt, edges = np.histogram(s, bins=bins)
    vmax = cnt.max() * 1.15
    bw = pw / bins

    out = [svg_open(w, h)]
    out.append(text(ml, 34, title, 17, INK, weight="700"))
    if unit:
        out.append(text(ml, 54, unit, 12, MUTED))

    for i in range(5):
        y = mt + ph - ph * i / 4
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')

    for i, c in enumerate(cnt):
        bh = ph * c / vmax
        out.append(f'<rect x="{ml+bw*i:.1f}" y="{mt+ph-bh:.1f}" width="{bw-1.5:.1f}" '
                   f'height="{bh:.1f}" fill="{ACC}" opacity="0.85"/>')

    for i in range(6):
        v = lo + (hi - lo) * i / 5
        x = ml + pw * i / 5
        out.append(text(x, mt + ph + 22, f"{v:.0f}", 11, MUTED, anchor="middle"))
    out.append(text(ml + pw / 2, mt + ph + 44, "元/㎡/月", 12, MUTED, anchor="middle"))

    med = series.median()
    xm = ml + pw * (med - lo) / (hi - lo)
    out.append(f'<line x1="{xm:.1f}" y1="{mt}" x2="{xm:.1f}" y2="{mt+ph}" '
               f'stroke="#b45309" stroke-width="1.6" stroke-dasharray="4 3"/>')
    out.append(text(xm + 6, mt + 16, f"中位数 {med:.0f}", 12, "#b45309", weight="700"))

    out.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(out))
    print("[图]", path)


# ---- 图 1：四城市中位租金单价 ----
city = (df.groupby("city")["unit_price"].median()
          .sort_values(ascending=False))
bar_chart(list(city.index), list(city.values),
          "四城市中位租金单价", f"{FIG_DIR}/01_city_rent.svg",
          unit="元/㎡/月　｜　北京与广州相差 1.93 倍")

# ---- 图 2：区县租金梯度（取样本量≥100 的区县，首尾各 8 个）----
cand = df.groupby("district")["unit_price"].agg(["size", "median"])
cand = cand[cand["size"] >= 100].sort_values("median", ascending=False)
top = cand.head(8)
bot = cand.tail(8)
sel = pd.concat([top, bot]).sort_values("median", ascending=False)
bar_chart(list(sel.index), list(sel["median"].values),
          "区县中位租金单价（样本量 ≥100 的区县，取最高与最低各 8 个）",
          f"{FIG_DIR}/02_district_rent.svg",
          unit="元/㎡/月　｜　静安 152.9 是从化 20.8 的 7.3 倍",
          w=880, h=450, highlight=set(range(8)))

# ---- 图 3：面积段与单价（非线性拐点）----
ab = df.groupby("area_bin", observed=True)["unit_price"].median()
bar_chart(list(ab.index.astype(str)), list(ab.values),
          "面积段与租金单价：90–120㎡ 最低，120㎡ 以上反弹",
          f"{FIG_DIR}/03_area_unitprice.svg",
          unit="元/㎡/月　｜　相关系数仅 -0.048，但分箱后规律清晰",
          h=430, highlight={3})

# ---- 图 4：回归系数效应 ----
coef_lbl = ["面积弹性", "卫生间/个", "精装", "地铁/公里", "房间数/间", "朝南"]
coef_val = [85.4, 14.8, 14.2, -9.7, -2.7, -3.7]   # 面积弹性按百分比展示
hbar_chart(coef_lbl, coef_val,
           "回归系数效应（相对基准组）", f"{FIG_DIR}/04_coef_effect.svg",
           unit="面积弹性为「面积涨1%时租金涨0.85%」，此处折算为百分比便于同图比较")

# ---- 图 5：单价分布 ----
hist_chart(df_all["unit_price"].dropna(), "租金单价分布（含合租样本）",
           f"{FIG_DIR}/05_price_dist.svg",
           unit="右偏分布，均值高于中位数，因此结论一律用中位数")

print("\n完成。共 5 张 SVG。")
