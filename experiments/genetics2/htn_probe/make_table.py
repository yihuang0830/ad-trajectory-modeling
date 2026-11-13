"""make_table.py -- 高血压扰动表（同 PPT 里基因表的版式：两层表头、浅蓝表头、细灰格线）。本地跑，只读 probe_htn_bp2.json。

    python make_table.py      # -> htn_table.png（贴幻灯片）、htn_table.xlsx（同一版式，可以自己再改、再截图）
"""
import glob
import json
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "probe_htn_bp2.json")))
R = P["rows"]

ROWS = [("htn", "高血压诊断 无 → 有（现在的改法）"),
        ("htn_rx", "诊断 + 降压药"),
        ("full", "诊断 + 降压药 + 血压 {bp}"),
        ("full160", "诊断 + 降压药 + 血压 160/90"),
        ("bp160", "只改血压 → 160/90"),
        ("rx", "只改降压药 无 → 有"),
        ("htn_clean", "只改诊断，只看原本没吃药、收缩压 < 140 的人")]
bp = P["cohorts"]["main"]["bp_onset"]
BP = f"{bp[0]:.0f}/{bp[1]:.0f}"


def y1(v):
    return "0.0" if abs(v) < 0.05 else f"{v:+.1f}"


def body():
    out = []
    for key, label in ROWS:
        m, t = R[key]["main"], R[key]["train"]
        p = m["pooled"]
        ye = p["years_earlier"]
        n_same = ye["n_seeds_positive"] if ye["value"] > 0 else 5 - ye["n_seeds_positive"]
        out.append([label.format(bp=BP), f"{p['10y']['risk_lo']:.1%}", f"{p['10y']['risk_hi']:.1%}",
                    y1(ye["value"]), y1(ye["ci95"][0]), y1(ye["ci95"][1]), f"{n_same}/5",
                    *[y1(m["by_visit"][str(v)]["years_earlier"]["value"]) for v in (2, 4, 6)],
                    y1(t["pooled"]["years_earlier"]["value"])])
    return out


# 表头：第一行 (文字, 起始列, 跨几列, 跨两行?)；第二行 (文字, 列)
COLW = [4.3, 1.05, 1.05, 1.35, 1.05, 1.05, 1.35, 0.95, 0.95, 0.95, 1.75]
TOP = [("扰动", 0, 1, True), ("10 年 AD 风险", 1, 2, False), ("AD 提前年数 [95% CI]", 3, 3, False),
       ("5 个 seed\n同向", 6, 1, True), ("第 2 / 4 / 6 次访视", 7, 3, False), ("train（模型见过）", 10, 1, False)]
SUB = [("扰动前", 1), ("扰动后", 2), ("点估计（年）", 3), ("CI 下限", 4), ("CI 上限", 5), ("第 2 次", 7), ("第 4 次", 8),
       ("第 6 次", 9), ("年", 10)]
HEAD_FILL, GRID, INK = "#d9e2f3", "#bfbfbf", "#000000"


def fonts():
    """PingFang SC（常规 / 半粗）：从系统的 PingFang.ttc 里取出来；找不到就用 Arial Unicode MS。"""
    ttc = glob.glob("/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/PingFang.ttc")
    if ttc:
        try:
            from fontTools.ttLib import TTCollection
            c = TTCollection(ttc[0])
            d = tempfile.mkdtemp()
            got = {}
            for f in c.fonts:
                nm = f["name"].getDebugName(4)
                if nm in ("PingFang SC Regular", "PingFang SC Semibold"):
                    p = os.path.join(d, nm.replace(" ", "_") + ".ttf")
                    f.save(p)
                    got[nm.split()[-1]] = fm.FontProperties(fname=p)
            if len(got) == 2:
                return got["Regular"], got["Semibold"]
        except Exception as e:  # noqa: BLE001
            print("PingFang 取不出来，用 Arial Unicode MS：", e)
    f = fm.FontProperties(family="Arial Unicode MS")
    return f, f


def png(rows, path):
    reg, bold = fonts()
    HH, RH, FS = 0.40, 0.36, 12.5
    xs = [sum(COLW[:i]) for i in range(len(COLW) + 1)]
    W, H = xs[-1], 2 * HH + RH * len(rows)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), W, 2 * HH, facecolor=HEAD_FILL, linewidth=0))
    line = lambda x0, y0, x1, y1, lw=0.8: ax.plot([x0, x1], [y0, y1], color=GRID, linewidth=lw, solid_capstyle="butt")  # noqa: E731
    for txt, c0, span, two in TOP:
        x0, x1 = xs[c0], xs[c0 + span]
        ax.text((x0 + x1) / 2, HH if two else HH / 2, txt, fontproperties=bold, fontsize=FS, color=INK, ha="center", va="center",
                linespacing=1.15)
        if not two:
            line(x0, HH, x1, HH)
    for txt, c in SUB:
        ax.text((xs[c] + xs[c + 1]) / 2, HH * 1.5, txt, fontproperties=bold, fontsize=FS, color=INK, ha="center", va="center")
    # 竖线：表头里跨两行的格子、和同一组里的子列分隔
    for i, x in enumerate(xs[1:-1], 1):
        sub_inner = any(c0 < i < c0 + span for _, c0, span, _ in TOP)
        line(x, HH if sub_inner else 0, x, H, 1.6 if i == 1 else 0.8)
    line(0, 2 * HH, W, 2 * HH, 1.6)
    for r, row in enumerate(rows):
        yc = 2 * HH + RH * (r + 0.5)
        for c, txt in enumerate(row):
            if c == 0:
                ax.text(xs[0] + 0.06, yc, txt, fontproperties=reg, fontsize=FS, color=INK, ha="left", va="center")
            else:
                ax.text((xs[c] + xs[c + 1]) / 2, yc, txt, fontproperties=reg,
                        fontsize=FS, color=INK, ha="center", va="center")
        line(0, 2 * HH + RH * (r + 1), W, 2 * HH + RH * (r + 1))
    line(0, 0, W, 0)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def xlsx(rows, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    ws = wb.active
    ws.title = "高血压扰动"
    thin = Side(style="thin", color="BFBFBF")
    bd = Border(left=thin, right=thin, top=thin, bottom=thin)
    fill = PatternFill("solid", fgColor=HEAD_FILL.lstrip("#").upper())
    hf, bf = Font(name="PingFang SC", bold=True, size=12), Font(name="PingFang SC", size=12)
    for txt, c0, span, two in TOP:
        ws.cell(1, c0 + 1, txt.replace("\n", ""))
        ws.merge_cells(start_row=1, start_column=c0 + 1, end_row=2 if two else 1, end_column=c0 + span)
    for txt, c in SUB:
        ws.cell(2, c + 1, txt)
    for r in (1, 2):
        for c in range(1, len(COLW) + 1):
            x = ws.cell(r, c)
            x.font, x.fill, x.border = hf, fill, bd
            x.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, row in enumerate(rows, 3):
        for c, txt in enumerate(row, 1):
            x = ws.cell(i, c, txt)
            x.font, x.border = bf, bd
            x.alignment = Alignment(horizontal="left" if c == 1 else "center", vertical="center")
    for c, w in enumerate(COLW, 1):
        ws.column_dimensions[get_column_letter(c)].width = w * 9.5
    for r in range(1, len(rows) + 3):
        ws.row_dimensions[r].height = 22
    wb.save(path)
    print("wrote", path)


if __name__ == "__main__":
    rows = body()
    png(rows, os.path.join(HERE, "htn_table.png"))
    xlsx(rows, os.path.join(HERE, "htn_table.xlsx"))
