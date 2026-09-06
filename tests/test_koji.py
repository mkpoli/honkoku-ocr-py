from honkoku_ocr.koji import raw_to_koji, raw_to_plain


def test_ruby_after_kanji_gets_slash():
    assert raw_to_koji("本<ruby>漢字<rt>かんじ</rt></ruby>") == "本／漢字（かんじ）"

def test_ruby_after_kana_no_slash():
    assert raw_to_koji("は<ruby>漢字<rt>かんじ</rt></ruby>") == "は漢字（かんじ）"

def test_wari_kaeri_okuri_tate():
    assert raw_to_koji("<WARI>右<WARI_SEP>左</WARI>") == "《割書：右｜左》"
    assert raw_to_koji("<WARI>右</WARI>") == "《割書：右》"
    assert raw_to_koji("子<KAERI>レ</KAERI>曰<OKURI>ク</OKURI>") == "子＿レ曰￣ク"
    assert raw_to_koji("a<TATE>b<BLOCK>c") == "aーbc"

def test_unknown_tags_removed_and_plain():
    assert raw_to_koji("x<foo>y</foo>") == "xy"
    assert raw_to_plain("本<ruby>漢字<rt>かんじ</rt></ruby>") == "本漢字かんじ"
    assert raw_to_koji("") == ""


def test_warigaki_keeps_its_marks_when_it_contains_kaeriten_or_okurigana():
    assert raw_to_koji("<WARI>或<OKURI>ハ</OKURI>従<WARI_SEP>四品</WARI>") == "《割書：或￣ハ従｜四品》"
    assert raw_to_koji("<WARI>読<OKURI>テ</OKURI><KAERI>二</KAERI>座敷<TATE>ト<KAERI>一</KAERI></WARI>") == "《割書：読￣テ＿二座敷ート＿一》"
    assert raw_to_koji("<WARI>右<WARI_SEP>左</WARI>") == "《割書：右｜左》"
