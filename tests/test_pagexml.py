from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from honkoku_ocr import pagexml
from honkoku_ocr.pipeline import LineResult, PageResult

NS = {"p": pagexml.NAMESPACE}


def result():
    lines = [LineResult(1, 100, 20, 30, 400, 0.9, "<ruby>漢<rt>かん</rt></ruby>", "漢（かん）", "漢かん"),
             LineResult(2, 40, 25, 30, 380, 0.8, "い", "い", "い")]
    return PageResult(1, 200, 500, 200, 500, 0, "v18", {}, lines, {}, [])


def test_page_xml_structure_and_order():
    data = pagexml.page_xml(result(), "page.jpg", created=datetime(2026, 9, 6, tzinfo=timezone.utc))
    root = ET.fromstring(data)
    page = root.find("p:Page", NS)
    assert page.get("imageFilename") == "page.jpg" and page.get("imageWidth") == "200" and page.get("imageHeight") == "500"
    lines = root.findall(".//p:TextLine", NS)
    assert [l.get("id") for l in lines] == ["l_1", "l_2"]
    assert lines[0].find("p:Coords", NS).get("points") == "100,20 130,20 130,420 100,420"
    assert lines[0].find("p:TextEquiv/p:Unicode", NS).text == "漢（かん）"
    assert root.find(".//p:TextRegion/p:TextEquiv/p:Unicode", NS).text == "漢（かん）\nい"
    assert root.find(".//p:Created", NS).text == "2026-09-06T00:00:00Z"
    assert root.find(".//p:RegionRefIndexed", NS).get("regionRef") == "r_1"


def test_plain_text_and_empty_page():
    root = ET.fromstring(pagexml.page_xml(result(), "p.jpg", plain=True))
    assert root.find(".//p:TextLine/p:TextEquiv/p:Unicode", NS).text == "漢かん"
    empty = ET.fromstring(pagexml.page_xml(PageResult(1, 10, 10, 10, 10, 0, "v18", {}, [], {}, []), "e.jpg"))
    assert empty.find(".//p:TextRegion", NS) is None
