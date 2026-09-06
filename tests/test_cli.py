from pathlib import Path
from honkoku_ocr import cli


def test_names_handle_same_parent_and_generated_name_collisions():
    files = [Path(p) for p in ('a/pages/p.jpg', 'b/pages/p.jpg', 'c/p.png',
                              'c/p__2.png', 'd/P.jpg')]
    names = cli._unique_names(files)
    assert len({name.casefold() for name in names.values()}) == len(files)
    assert names[Path('c/p__2.png')] == 'p__2'


def test_cli_deduplicates_inputs_and_ignores_image_named_directories(tmp_path, monkeypatch):
    page = tmp_path / 'page.png'
    page.touch()
    (tmp_path / 'directory.jpg').mkdir()
    seen = []

    class FakeOCR:
        def __init__(self, *args):
            pass

        def run(self, path):
            seen.append(path)
            return []

    monkeypatch.setattr(cli, 'OCR', FakeOCR)
    output = tmp_path / 'out'
    assert cli.main([str(tmp_path), str(page), '-o', str(output)]) == 0
    assert seen == [page]
    assert (output / 'page.json').is_file()
