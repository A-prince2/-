# -*- coding: utf-8 -*-
"""
============================================================
城市租房数据采集 · 代码骨架（Skeleton，非成品脚本）
============================================================

【这是什么】
    一份供你动手填写的采集框架。它已经搭好了分层结构、网络保护、
    合规检查和 CSV 落盘，但**解析层是空的**——你必须自己从页面里
    找出 CSS 选择器填进去，程序才能真正产出数据。

    换句话说：直接运行它，你会看到"未解析到任何数据"，
    这不是 bug，是设计意图。

【使用前提 · 请先读】
    1. 使用前必须先确认目标站点的 robots 协议允许抓取。
       调用 can_fetch() 检查，返回 False 就换目标，不要绕过。
    2. 本项目定位为学习研究，仅抓取公开页面、串行请求、
       间隔 2-5 秒、总量控制在 2000-5000 条。
    3. 不要采集房东/中介的姓名、手机、微信等个人信息。
    4. 不要破解加密参数、不要用代理池规避封禁。
    5. 更推荐的做法：先用公开数据集（如和鲸 CC0 数据集）走通
       分析全流程，再用本骨架练解析能力。

【依赖】
    pip install requests beautifulsoup4

【调试顺序】（别跳步，跳步会更慢）
    第 1 步  只请求第 1 页 → 存 HTML 快照 → 确认能拿到完整页面
    第 2 步  在快照上只抽 rent 一个字段 → print 前 3 条 → 人工核对
    第 3 步  补齐字段 → 抽 1 页 → 存 CSV → Excel 打开检查列是否错位
    第 4 步  加翻页 → 跑到第 3 页就停 → 检查有无重复
    第 5 步  放开上限 → 全量小规模采集

    跳过第 2 步是新手最常见的错误：直接跑全量，结果二十个字段全空，
    你根本不知道是网络问题、页面结构变了，还是自己选择器写错了。

【目录约定】
    project/
    ├── collector_skeleton.py   本文件
    ├── data/rent_raw.csv       采集结果（追加写，不要手改）
    ├── data/progress.json      断点进度
    └── debug/page_001.html     HTML 快照（调试用）
============================================================
"""

import csv
import json
import os
import random
import re
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


# ============================================================
# [1] 配置区 —— 所有常量集中在这里，调参只改这一块
# ============================================================

# 目标站点。先自行核实 robots 是否允许，再填这里。
BASE_URL = "https://example.com"          # TODO: 换成已确认合规的目标站
LIST_PATH = "/rent/pg{page}/"             # TODO: 按目标站分页规则修改

# 表头字段顺序。顺序一旦定下就不要改，否则追加写会错列。
FIELDS = [
    "house_id", "city", "district", "business_area", "community",
    "rent", "area", "layout_raw", "orientation", "floor_raw",
    "decoration", "elevator", "collect_date", "source_url", "page_no",
]

# 字段 → CSS 选择器 的映射表。
# 为什么要用字典而不是写死二十行代码？
# 因为页面改版时你只需要改这张表，解析逻辑一行都不用动。
FIELD_SELECTORS = {
    "house_id":      "",   # TODO: 例如 "[data-house-id]"
    "district":      "",   # TODO
    "business_area": "",   # TODO
    "community":     "",   # TODO
    "rent":          "",   # TODO  抽数字，如 ".price"
    "area":          "",   # TODO  抽数字
    "layout_raw":    "",   # TODO  如 "2室1厅"
    "orientation":   "",   # TODO
    "floor_raw":     "",   # TODO  如 "中楼层/共18层"
    "decoration":    "",   # TODO
    "elevator":      "",   # TODO
}

# 列表页的父容器与单个房源条目的选择器 —— 这两个是最关键的，必须自己找
LIST_CONTAINER_SELECTOR = ""   # TODO
ITEM_SELECTOR = ""             # TODO

# 请求与限速参数
USER_AGENT = "StudentResearchBot/1.0 (learning project; contact: your@email.com)"
REQUEST_TIMEOUT = (5, 15)      # (连接超时, 读取超时)，单位秒
SLEEP_RANGE = (2, 5)           # 每次翻页后的随机等待区间，单位秒
MAX_PAGES = 20                 # 单次运行最多翻多少页，压低上限=降低风险
COLLECT_DATE = time.strftime("%Y-%m-%d")

# 路径
DATA_DIR = "data"
DEBUG_DIR = "debug"
CSV_PATH = os.path.join(DATA_DIR, "rent_raw.csv")
PROGRESS_PATH = os.path.join(DATA_DIR, "progress.json")


# ============================================================
# [2] 合规检查 —— 动手前先跑这个
# ============================================================

def can_fetch(url, user_agent=USER_AGENT):
    """
    检查目标 URL 是否被 robots.txt 允许抓取。

    这是整份代码里最该先跑的函数。返回 False 就停止，
    不要用改 User-Agent 的方式规避——那等于隐去真实身份，
    性质从"抓取"变成"绕过访问控制"，风险完全不同。
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception as exc:
        # 读不到 robots 时保守处理：视为未授权
        print(f"[!] 无法读取 {robots_url}: {exc}")
        return False
    return rp.can_fetch(user_agent, url)


# ============================================================
# [3] 请求层 —— 只管拿到 HTML，不关心内容
# ============================================================

def build_session():
    """
    构造一个带默认头的 Session。

    用 Session 而不是每次 requests.get，好处是复用 TCP 连接，
    对服务器更友好，也省去重复握手开销。
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,          # 如实标明身份，不伪装浏览器
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    return session


def fetch(session, url, retry=3):
    """
    请求单个页面，带超时、重试与状态码判断。

    三个必须做的保护：
      1) timeout —— 没有超时的请求可能永久挂住，你会误以为程序卡死
      2) 状态码显式判断 —— 不要假设一定返回 200
      3) 403/429 立即停止 —— 这是明确的"请停下"信号，不要重试
    """
    for attempt in range(retry):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 200:
                # 编码处理：网页头部有时声明错误，apparent_encoding 作为兜底
                resp.encoding = resp.apparent_encoding
                return resp

            if resp.status_code in (403, 429):
                print(f"[!] 收到 {resp.status_code}，对方拒绝或限流，停止采集")
                return None

            print(f"[!] 状态码 {resp.status_code}，第 {attempt + 1} 次尝试")

        except requests.RequestException as exc:
            print(f"[!] 请求异常（第 {attempt + 1} 次）: {exc}")

        # 退避重试：间隔逐次翻倍，别用固定间隔猛冲
        time.sleep(2 ** attempt)

    return None


def save_snapshot(html, page_no, dir_=DEBUG_DIR):
    """
    把 HTML 存成本地快照。

    为什么必须做这件事：解析逻辑写错时，你不需要重新请求服务器，
    直接用快照重跑解析即可。既省时间，也减少对目标站的压力。
    """
    os.makedirs(dir_, exist_ok=True)
    path = os.path.join(dir_, f"page_{page_no:03d}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ============================================================
# [4] 解析层 —— 需要你自己填的部分
# ============================================================

def to_number(text):
    """
    从带单位的文本里抽出数字。

    "3200元/月" → 3200.0
    "89.5㎡"    → 89.5
    "1,200"     → 1200.0
    "--" / None → None
    """
    if not text:
        return None
    cleaned = str(text).replace(",", "").strip()
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def parse_list_page(html, page_no):
    """
    把一页 HTML 解析成若干条字典记录。

    >>> 这里是你要动脑的地方，思路按顺序如下： <<<

    第 1 步：在浏览器里打开 debug/page_001.html（不是打开真实网站），
             按 F12 检查元素。快照才是你手上这份 HTML。

    第 2 步：确认数据是不是静态渲染的。
             如果快照里搜不到租金数字，说明数据靠 JS 异步加载，
             bs4 拿不到 —— 此时应换目标站点，不要硬啃。

    第 3 步：找列表父容器。整个房源列表一定有一个共同的父元素，
             先 select_one 定位它。

    第 4 步：在父容器内 select 出每个房源条目，循环处理。

    第 5 步：每个字段单独 try，抽不到就置 None。
             单条脏数据绝不能中断整页解析。

    提示：选择器尽量写得"浅"一些，依赖更稳定的 class 或属性，
          不要写一长串精确到层级的路径 —— 页面稍微改版就全废了。
    """
    soup = BeautifulSoup(html, "html.parser")

    if not LIST_CONTAINER_SELECTOR or not ITEM_SELECTOR:
        print("[!] 解析选择器还没填，这是骨架的正常状态")
        print("    请先从 debug/page_001.html 里找到容器与条目的 CSS 选择器，")
        print("    填入 FIELD_SELECTORS / LIST_CONTAINER_SELECTOR / ITEM_SELECTOR")
        return []

    container = soup.select_one(LIST_CONTAINER_SELECTOR)
    if container is None:
        print(f"[!] 第 {page_no} 页没找到列表容器，可能页面结构变了或被拦截")
        return []

    rows = []
    for item in container.select(ITEM_SELECTOR):
        row = {"page_no": page_no, "collect_date": COLLECT_DATE}

        # 逐字段抽取。注意这里没有一处会让异常冒到循环外。
        for field, selector in FIELD_SELECTORS.items():
            if not selector:
                row[field] = None
                continue
            try:
                node = item.select_one(selector)
                row[field] = node.get_text(strip=True) if node else None
            except Exception:
                row[field] = None

        # 数值型字段统一转成数字，方便后续计算
        row["rent"] = to_number(row.get("rent"))
        row["area"] = to_number(row.get("area"))

        # 记录来源地址，保证可追溯
        row["source_url"] = item.get("href") or None

        rows.append(row)

    return rows


# ============================================================
# [5] 存储层 —— 只管把记录写进 CSV
# ============================================================

def save_rows(rows, path=CSV_PATH):
    """
    追加写入 CSV。四个参数一个都不能少：

      encoding="utf-8-sig"  带 BOM，Excel 双击打开中文不乱码
      newline=""            Windows 下不加会每行多一个空行
      mode="a"              追加。用 "w" 会覆盖，中断后前面采的全没了
      fieldnames=FIELDS     固定顺序，否则不同记录列会错位
    """
    if not rows:
        return 0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new_file = not os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            extrasaction="ignore",   # 解析层多抽了字段也不会崩
        )
        if is_new_file:
            writer.writeheader()     # 只在首次写表头
        for row in rows:
            writer.writerow({k: row.get(k) for k in FIELDS})

    return len(rows)


def load_progress():
    """读取断点进度，返回已完成的页码；没有记录则返回 0。"""
    try:
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("last_page", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def save_progress(page_no):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_page": page_no}, f, ensure_ascii=False)


# ============================================================
# [6] 编排层 —— 翻页循环与整体节奏
# ============================================================

def main():
    # ---- 动手前的合规检查，不通过就结束 ----
    first_url = BASE_URL + LIST_PATH.format(page=1)
    if not can_fetch(first_url):
        print(f"[x] robots.txt 不允许抓取 {first_url}，程序退出。")
        print("    请更换目标站点，不要尝试绕过。")
        return

    session = build_session()
    start_page = load_progress() + 1
    end_page = start_page + MAX_PAGES      # 单次运行的上限，别调太高

    print(f"[*] 从第 {start_page} 页开始，最多 {MAX_PAGES} 页")

    for page_no in range(start_page, end_page):
        url = BASE_URL + LIST_PATH.format(page=page_no)
        print(f"[>] 第 {page_no} 页 {url}")

        resp = fetch(session, url)
        if resp is None:
            print("[x] 拿不到页面，停止。请检查是否被限流。")
            break

        # 存快照是调试的关键：解析出错时可以离线重跑
        save_snapshot(resp.text, page_no)

        rows = parse_list_page(resp.text, page_no)
        if not rows:
            print("[x] 本页未解析到数据。若连续两页如此，停止。")
            break

        count = save_rows(rows)
        save_progress(page_no)          # 顺序很重要：先落 CSV，再记进度
        print(f"[+] 写入 {count} 条，累计进度到第 {page_no} 页")

        # 随机间隔。不要用固定值，也不要往下调——这个数字是给对方的缓冲。
        time.sleep(random.uniform(*SLEEP_RANGE))

    print("[*] 结束。检查 data/rent_raw.csv，用 Excel 打开确认列没错位。")


if __name__ == "__main__":
    main()
