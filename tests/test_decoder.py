import numpy as np
import pytest
from PIL import Image

from honkoku_ocr import recognizer
from honkoku_ocr.recognizer import Recognizer


def fake_recognizer(monkeypatch, tokens, max_tokens=192):
    monkeypatch.setattr(recognizer, 'to_pixel', lambda *a: np.zeros((1, 3, 1, 1), np.float32))
    class Encoder:
        def run(self, *args):
            return [np.zeros((1, 2, 3), np.float32)]
    class Decoder:
        def __init__(self):
            self.calls = 0
        def run(self, names, feeds):
            if self.calls:
                assert feeds['past_0'].item() == self.calls - 1
                assert feeds['input_ids'].item() == tokens[self.calls - 1]
            logits = np.zeros((1, 1, 40), np.float32)
            logits[0, 0, tokens[self.calls]] = 1
            output = [logits, np.array([self.calls])]
            self.calls += 1
            return output
    decoder = Decoder()
    rec = object.__new__(Recognizer)
    rec.enc, rec.enc_in = Encoder(), 'pixels'
    rec.pre = rec.step = decoder
    rec.img_h = rec.img_w = 1
    rec.past, rec.present, rec.pre_out = ['past_0'], ['present_0'], ['logits', 'present_0']
    rec.max_tokens = max_tokens
    rec.vocab = [''] * 5 + [str(i) for i in range(5, 40)]
    return rec, decoder


@pytest.mark.parametrize('tokens,limit,expected,reason', [
    ([3], 192, [], 'eos'), ([5, 6, 3], 192, [5, 6], 'eos'),
    ([5, 6, 7], 3, [5, 6, 7], 'max_tokens'),
    ([5] * 12, 192, [5], 'repetition'),
    ([5, 6] * 6, 192, [5, 6], 'repetition'),
])
def test_decoder_cache_progression_and_stop_reasons(monkeypatch, tokens, limit, expected, reason):
    rec, decoder = fake_recognizer(monkeypatch, tokens, limit)
    ids, actual_reason, timings = rec._generate(Image.new('RGB', (1, 1)))
    assert ids == expected and actual_reason == reason
    assert decoder.calls == len(tokens)
    assert set(timings) == {'preprocess', 'encoder', 'prefill', 'decode'}
