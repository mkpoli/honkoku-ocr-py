"""PAGE XML（PRImA、2019-07-15スキーマ）への書き出し。

1ページを1つのTextRegionとし、行ごとにTextLine（矩形のCoords、TextEquiv）を読み順に並べる。
座標はページ記録と同じくEXIFの向きを反映した元画像のもの。TextEquivにはKoji記法か素テキストを入れる。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from .output import package_version

if TYPE_CHECKING:
    from .pipeline import PageResult

NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
SCHEMA = f"{NAMESPACE} {NAMESPACE}/pagecontent.xsd"


def _points(x: float, y: float, w: float, h: float) -> str:
    x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    return f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"


def page_xml(result: PageResult, image: str, *, plain: bool = False, created: datetime | None = None) -> bytes:
    """`PageResult`をPAGE XMLのバイト列にする。`image`は記録するファイル名。"""
    ET.register_namespace("", NAMESPACE)
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root = ET.Element(f"{{{NAMESPACE}}}PcGts", {"{http://www.w3.org/2001/XMLSchema-instance}schemaLocation": SCHEMA})
    stamp = (created or datetime.now(timezone.utc)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    meta = ET.SubElement(root, f"{{{NAMESPACE}}}Metadata")
    ET.SubElement(meta, f"{{{NAMESPACE}}}Creator").text = f"honkoku-ocr-py {package_version()} (kuzushiji {result.model})"
    ET.SubElement(meta, f"{{{NAMESPACE}}}Created").text = stamp
    ET.SubElement(meta, f"{{{NAMESPACE}}}LastChange").text = stamp
    page = ET.SubElement(root, f"{{{NAMESPACE}}}Page", {"imageFilename": image, "imageWidth": str(result.width),
                                                        "imageHeight": str(result.height)})
    if result.lines:
        order = ET.SubElement(page, f"{{{NAMESPACE}}}ReadingOrder")
        group = ET.SubElement(order, f"{{{NAMESPACE}}}OrderedGroup", {"id": "ro_1"})
        ET.SubElement(group, f"{{{NAMESPACE}}}RegionRefIndexed", {"index": "0", "regionRef": "r_1"})
        xs = [line.x for line in result.lines] + [line.x + line.width for line in result.lines]
        ys = [line.y for line in result.lines] + [line.y + line.height for line in result.lines]
        region = ET.SubElement(page, f"{{{NAMESPACE}}}TextRegion", {"id": "r_1", "type": "paragraph",
                                                                     "custom": "readingDirection {vertical-rl}"})
        ET.SubElement(region, f"{{{NAMESPACE}}}Coords", {"points": _points(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))})
        for line in result.lines:
            element = ET.SubElement(region, f"{{{NAMESPACE}}}TextLine", {
                "id": f"l_{line.reading_order}", "custom": f"readingOrder {{index:{line.reading_order - 1};}}"})
            ET.SubElement(element, f"{{{NAMESPACE}}}Coords", {"points": _points(line.x, line.y, line.width, line.height)})
            equiv = ET.SubElement(element, f"{{{NAMESPACE}}}TextEquiv", {"conf": f"{line.detection_confidence:.4f}"})
            ET.SubElement(equiv, f"{{{NAMESPACE}}}Unicode").text = line.plain if plain else line.koji
        equiv = ET.SubElement(region, f"{{{NAMESPACE}}}TextEquiv")
        ET.SubElement(equiv, f"{{{NAMESPACE}}}Unicode").text = "\n".join(line.plain if plain else line.koji for line in result.lines)
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
