"""デコーダの特殊トークン列と Koji 記法（みんなで翻刻の記法）の相互変換。

<ruby>親<rt>よみ</rt></ruby> → 親（よみ）  （直前が漢字なら ／親（よみ））
<WARI>右<WARI_SEP>左</WARI>          → 《割書：右｜左》
<KAERI>レ</KAERI>                     → ＿レ
<OKURI>かな</OKURI>                   → ￣かな
<TATE>                                → ー
"""
from __future__ import annotations

import re

_RUBY = re.compile(r"<ruby>([^<]*)<rt>([^<]*)</rt>(?:<rt2>([^<]*)</rt2>)?</ruby>")
_WARI = re.compile(r"<WARI>([^<]*?)(?:<WARI_SEP>([^<]*?))?</WARI>")
_KAERI = re.compile(r"<KAERI>([^<]*)</KAERI>")
_OKURI = re.compile(r"<OKURI>([^<]*)</OKURI>")
_TAG = re.compile(r"<[^>]+>")

def _is_kanji(cp: int) -> bool:
    return 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2FA1F

def raw_to_koji(raw: str) -> str:
    if not raw:
        return ""
    def ruby(m: re.Match) -> str:
        slash = "／" if m.start() > 0 and _is_kanji(ord(raw[m.start() - 1])) else ""
        base, rt, rt2 = m.group(1), m.group(2), m.group(3)
        return f"{slash}{base}（{rt}｜{rt2}）" if rt2 else f"{slash}{base}（{rt}）"
    s = _RUBY.sub(ruby, raw)
    # 返り点・送り仮名・縦点を先に記号へ直してから割書を変換する。割書の中にこれらのタグが
    # 残っていると割書の正規表現が合わず、《割書：》の印が落ちる。みんなで翻刻の翻刻文では
    # 割書の約6%が返り点か送り仮名を含む。
    s = _KAERI.sub(r"＿\1", s)
    s = _OKURI.sub(r"￣\1", s)
    s = s.replace("<TATE>", "ー").replace("<BLOCK>", "")
    s = _WARI.sub(lambda m: f"《割書：{m.group(1)}｜{m.group(2)}》" if m.group(2) is not None else f"《割書：{m.group(1)}》", s)
    return _TAG.sub("", s)

def raw_to_plain(raw: str) -> str:
    """タグを除いた素のテキスト（ふりがなの読みも連結される）。"""
    return _TAG.sub("", raw or "")
