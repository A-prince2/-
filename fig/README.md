# 图表

由 `make_figs.py` 生成，格式为 SVG —— 不依赖 matplotlib，GitHub 上直接渲染，任意缩放不失真。

| 文件 | 内容 | 报告中的位置 |
|---|---|---|
| `05_price_dist.svg` | 租金单价分布直方图 + 中位数标注 | 4.1 单变量分布 |
| `01_city_rent.svg` | 四城市中位租金单价对比 | 4.2 城市梯度 |
| `02_district_rent.svg` | 区县租金梯度（样本量 ≥100，取首尾各 8 个） | 4.2 区县梯度 |
| `03_area_unitprice.svg` | 面积段与单价，标注 90–120㎡ 最低点 | 4.5 面积非线性 |
| `04_coef_effect.svg` | 回归系数效应（正负双向条形） | 5.3 回归系数 |

## 重新生成

```bash
python make_figs.py
```

需要先跑过 `cleaning_eda.py`，因为图表依赖 `data/rent_clean.csv`。

## 说明

- 图表用**中位数**而不是均值——租金右偏（偏度 4.41），均值会被少量高价房源拉高。
- `05_price_dist.svg` 用 0.5% / 99.5% 分位裁剪了横轴，否则极端值（16 万元/月）会把所有柱子压扁。
- 若你的环境装了 matplotlib，也可以在 `cleaning_eda.py` 里直接生成 PNG 版本，两者不冲突。
