"""XY-Cut による読み順付与。縦書きでは右の段から左へ、段内は上から下。

NDL古典籍OCR-Lite の block_xy_cut と honkoku-ocr-web の reading-order.ts に従う。
"""
from __future__ import annotations
from dataclasses import dataclass, field

GRID = 100

@dataclass
class _Node:
    x0: int; y0: int; x1: int; y1: int
    children: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    n: int = 0
    nv: int = 0
    xsplit: bool = False

def order(boxes: list[tuple[float, float, float, float]]) -> list[int]:
    """boxes = [(x, y, w, h)] → 読み順 (0 始まり) を各 box に対して返す。"""
    if not boxes:
        return []
    if len(boxes) == 1:
        return [0]
    raw = [(x, y, x + w, y + h) for x, y, w, h in boxes]
    norm, w, h = _normalize(raw)
    table = [[0] * w for _ in range(h)]
    for x0, y0, x1, y1 in norm:
        for yy in range(y0, min(y1, h)):
            row = table[yy]
            for xx in range(x0, min(x1, w)):
                row[xx] = 1
    root = _Node(0, 0, w, h)
    _cut(table, root)
    _assign(root, norm)
    _sort(root, norm)
    ranks = [-1] * len(raw)
    _rank(root, ranks, 0)
    return ranks

def _normalize(b):
    xmin = min(v[0] for v in b); ymin = min(v[1] for v in b)
    xmax = max(v[2] for v in b); ymax = max(v[3] for v in b)
    wp, hp = xmax - xmin, ymax - ymin
    if wp == 0 or hp == 0:
        return [[0, 0, 1, 1] for _ in b], 2, 2
    portrait = hp >= wp
    xg = GRID * (wp / hp) if portrait else GRID
    yg = GRID if portrait else GRID * (hp / wp)
    import math
    w, h = math.ceil(xg) + 1, math.ceil(yg) + 1
    out = []
    for x0, y0, x1, y1 in b:
        nx0 = max(0, math.floor((x0 - xmin) * xg / wp)); ny0 = max(0, math.floor((y0 - ymin) * yg / hp))
        nx1 = min(w - 1, math.ceil((x1 - xmin) * xg / wp)); ny1 = min(h - 1, math.ceil((y1 - ymin) * yg / hp))
        out.append([nx0, ny0, max(nx0 + 1, nx1), max(ny0 + 1, ny1)])
    return out, w, h

def _hist(table, x0, y0, x1, y1):
    xh = [0] * (x1 - x0); yh = [0] * (y1 - y0)
    for y in range(y0, y1):
        row = table[y]
        for x in range(x0, x1):
            v = row[x]
            xh[x - x0] += v; yh[y - y0] += v
    return xh, yh

def _min_span(hist):
    if len(hist) <= 1:
        return 0, len(hist), 0.0
    lo, hi = min(hist), max(hist)
    best_s = best_e = best_len = 0
    start = -1
    for i in range(len(hist) + 1):
        if i < len(hist) and hist[i] == lo:
            if start == -1:
                start = i
        else:
            if start != -1:
                if i - start > best_len:
                    best_len, best_s, best_e = i - start, start, i
                start = -1
    return best_s, best_e, (-lo / hi if hi > 0 else 0.0)

def _cut(table, node):
    x0, y0, x1, y1 = node.x0, node.y0, node.x1, node.y1
    if x0 >= x1 or y0 >= y1:
        return
    xh, yh = _hist(table, x0, y0, x1, y1)
    xb0, xe0, xv = _min_span(xh); yb0, ye0, yv = _min_span(yh)
    xb, xe, yb, ye = xb0 + x0, xe0 + x0, yb0 + y0, ye0 + y0
    if x0 == xb and x1 == xe and y0 == yb and y1 == ye:
        return
    if yv < xv:
        _split_x(table, node, xb, xe)
    elif xv < yv:
        _split_y(table, node, yb, ye)
    elif (xe - xb) < (ye - yb):
        _split_y(table, node, yb, ye)
    else:
        _split_x(table, node, xb, xe)

def _split_x(table, p, g0, g1):
    p.xsplit = True
    _add(table, p, p.x0, p.y0, g0, p.y1); _add(table, p, g0, p.y0, g1, p.y1); _add(table, p, g1, p.y0, p.x1, p.y1)

def _split_y(table, p, g0, g1):
    p.xsplit = False
    _add(table, p, p.x0, p.y0, p.x1, g0); _add(table, p, p.x0, g0, p.x1, g1); _add(table, p, p.x0, g1, p.x1, p.y1)

def _add(table, p, x0, y0, x1, y1):
    if x0 >= x1 or y0 >= y1:
        return
    if (x0, y0, x1, y1) == (p.x0, p.y0, p.x1, p.y1):
        return
    c = _Node(x0, y0, x1, y1); p.children.append(c); _cut(table, c)

def _leaves(n):
    return [n] if not n.children else [l for c in n.children for l in _leaves(c)]

def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

def _assign(root, boxes):
    leaves = _leaves(root)
    lb = [(l.x0, l.y0, l.x1, l.y1) for l in leaves]
    for i, b in enumerate(boxes):
        best, best_v = 0, -1.0
        for j, l in enumerate(lb):
            v = _iou(b, l)
            if v > best_v:
                best_v, best = v, j
        leaves[best].lines.append(i)

def _vertical(n):
    return n.n < n.nv * 2

def _sort(node, boxes):
    if node.lines:
        node.n = len(node.lines)
        node.nv = sum(1 for i in node.lines if (boxes[i][2] - boxes[i][0]) < (boxes[i][3] - boxes[i][1]))
        if len(node.lines) > 1:
            if _vertical(node):
                node.lines.sort(key=lambda i: (-boxes[i][0], boxes[i][1]))
            else:
                node.lines.sort(key=lambda i: (boxes[i][1], boxes[i][0]))
    else:
        for c in node.children:
            n, v = _sort(c, boxes); node.n += n; node.nv += v
        if node.xsplit and _vertical(node):
            node.children.reverse()
    return node.n, node.nv

def _rank(node, ranks, r):
    for i in node.lines:
        ranks[i] = r; r += 1
    for c in node.children:
        r = _rank(c, ranks, r)
    return r
